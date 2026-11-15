//! Inference worker: one thread that runs the model, fed by a bounded queue.
//!
//! Request handlers never run the model. They take an [`InferenceTicket`]
//! (a place in the queue, refused at once when the queue is full), prepare
//! their input, hand it over with [`InferenceTicket::run`] and wait for the
//! result on a oneshot channel. The dynamic batcher hands over whole
//! batches the same way.
//!
//! A single OS thread takes jobs off the queue in arrival order and runs
//! them on the pipeline, one job at a time. Before it runs a job it drops
//! every request whose caller has stopped waiting or whose deadline has
//! passed, so work nobody will read is never started.
//!
//! The worker, not the handler, owns a job from the moment it is queued:
//! the prepared input, the deadline and the reply channel all travel in
//! the [`Job`]. A later change that runs several generations in one loop
//! can therefore take more jobs off the queue between token steps (a
//! non-blocking variant of [`Shared::next_job`]) and answer each one as it
//! finishes, without touching the handlers.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use hephaestus_core::{PipelineKind, PipelineOutput, PreparedData};
use tokio::sync::{RwLock, oneshot};

use crate::batcher::BatchRequest;
use crate::error::ApiError;
use crate::metrics::StageTimer;

/// Requests that may wait for the model when no capacity is configured.
pub const DEFAULT_QUEUE_CAPACITY: usize = 1024;

/// Seconds a client refused for a full queue is told to wait
/// (`Retry-After`).
pub const QUEUE_FULL_RETRY_AFTER_SECS: u64 = 1;

/// Reply channel of one request.
type Reply = oneshot::Sender<Result<PipelineOutput, ApiError>>;

/// One request's place among the waiting requests.
///
/// Counted in the queue depth from admission until the worker starts the
/// request or the request is given up; dropping the slot frees the place.
struct Slot {
    shared: Arc<Shared>,
    admitted_at: Instant,
}

impl Slot {
    /// The worker is starting this request: record how long it waited.
    fn started(self) {
        metrics::histogram!(
            "hephaestus_inference_queue_wait_seconds",
            "model_id" => self.shared.model_id.clone(),
        )
        .record(self.admitted_at.elapsed().as_secs_f64());
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.shared.waiting.fetch_sub(1, Ordering::SeqCst);
        self.shared.depth_gauge().decrement(1.0);
    }
}

/// One request inside a job.
struct Request {
    prepared: PreparedData,
    /// When the caller stops waiting; `None` for callers without a timeout.
    deadline: Option<Instant>,
    reply: Reply,
    slot: Slot,
}

impl Request {
    /// Take the request off the waiting list as the worker reaches it.
    ///
    /// Returns the input and reply channel of a request that should run,
    /// and `None` for one whose caller is gone or whose deadline has
    /// passed; the latter is answered with [`ApiError::Timeout`].
    fn start(self, now: Instant) -> Option<(PreparedData, Reply)> {
        let Self {
            prepared,
            deadline,
            reply,
            slot,
        } = self;
        let shared = Arc::clone(&slot.shared);
        if reply.is_closed() {
            shared.count_dropped("abandoned");
            return None;
        }
        if deadline.is_some_and(|deadline| now >= deadline) {
            shared.count_dropped("expired");
            let _ = reply.send(Err(ApiError::Timeout));
            return None;
        }
        slot.started();
        Some((prepared, reply))
    }
}

/// A unit of work for the worker: one request, or one batch.
struct Job {
    id: u64,
    requests: Vec<Request>,
    /// Run through `execute_batch` (the dynamic batcher's jobs) instead of
    /// `execute`.
    batched: bool,
    /// Records the `inference` stage duration of a single request.
    timer: Option<StageTimer>,
    /// Span of the submitting task; the job runs inside it.
    span: tracing::Span,
    /// Dropped when the job is finished or discarded, which wakes a
    /// submitter waiting for the whole job.
    _done: oneshot::Sender<()>,
}

/// Jobs waiting for the worker.
struct Queue {
    jobs: VecDeque<Job>,
    /// Set at shutdown: nothing more is accepted and the worker stops
    /// after the job it is running.
    closed: bool,
    next_id: u64,
}

/// State shared by the worker thread, its handle and every ticket.
struct Shared {
    model_id: Arc<str>,
    /// Most requests that may wait at one time.
    capacity: AtomicUsize,
    /// Requests admitted and not yet started: holding a ticket, or queued.
    waiting: AtomicUsize,
    queue: Mutex<Queue>,
    /// Signalled when a job is queued or the queue is closed.
    wake: Condvar,
    /// Test seam: runs on the worker thread just before each job's model
    /// run. Lets a test stand in for a slow model.
    #[cfg(test)]
    before_inference: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl Shared {
    fn lock_queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn depth_gauge(&self) -> metrics::Gauge {
        metrics::gauge!(
            "hephaestus_inference_queue_depth",
            "model_id" => self.model_id.clone(),
        )
    }

    fn count_rejected(&self, reason: &'static str) {
        metrics::counter!(
            "hephaestus_inference_jobs_rejected_total",
            "model_id" => self.model_id.clone(),
            "reason" => reason,
        )
        .increment(1);
    }

    fn count_dropped(&self, reason: &'static str) {
        metrics::counter!(
            "hephaestus_inference_jobs_dropped_total",
            "model_id" => self.model_id.clone(),
            "reason" => reason,
        )
        .increment(1);
    }

    /// Count one more waiting request. With `bounded`, refuse when the
    /// queue is at capacity.
    fn take_slot(self: &Arc<Self>, bounded: bool) -> Option<Slot> {
        if bounded {
            let capacity = self.capacity.load(Ordering::SeqCst);
            self.waiting
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |waiting| {
                    (waiting < capacity).then_some(waiting + 1)
                })
                .ok()?;
        } else {
            self.waiting.fetch_add(1, Ordering::SeqCst);
        }
        self.depth_gauge().increment(1.0);
        Some(Slot {
            shared: Arc::clone(self),
            admitted_at: Instant::now(),
        })
    }

    /// Queue a job. Returns its ID and a channel that closes when the job
    /// is finished or discarded.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::NotReady`] once the queue is closed.
    fn push(
        &self,
        requests: Vec<Request>,
        batched: bool,
        timer: Option<StageTimer>,
    ) -> Result<(u64, oneshot::Receiver<()>), ApiError> {
        let (done, finished) = oneshot::channel();
        let mut queue = self.lock_queue();
        if queue.closed {
            drop(queue);
            self.count_rejected("shutting_down");
            return Err(ApiError::NotReady);
        }
        let id = queue.next_id;
        queue.next_id += 1;
        queue.jobs.push_back(Job {
            id,
            requests,
            batched,
            timer,
            span: tracing::Span::current(),
            _done: done,
        });
        drop(queue);
        self.wake.notify_one();
        Ok((id, finished))
    }

    /// Take a job out of the queue because its caller stopped waiting.
    /// Does nothing if the worker already took it.
    fn abandon(&self, id: u64) {
        let job = {
            let mut queue = self.lock_queue();
            let position = queue.jobs.iter().position(|job| job.id == id);
            position.and_then(|position| queue.jobs.remove(position))
        };
        if job.is_some() {
            self.count_dropped("abandoned");
        }
    }

    /// Block until a job is queued. Returns `None` once the queue is
    /// closed.
    fn next_job(&self) -> Option<Job> {
        let mut queue = self.lock_queue();
        loop {
            if queue.closed {
                return None;
            }
            if let Some(job) = queue.jobs.pop_front() {
                return Some(job);
            }
            queue = self
                .wake
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Run the test seam, if one is installed.
    fn before_inference(&self) {
        #[cfg(test)]
        {
            let hook = self
                .before_inference
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if let Some(hook) = hook {
                hook();
            }
        }
    }

    /// Stop accepting jobs and fail the queued ones.
    fn close(&self) {
        let queued: Vec<Job> = {
            let mut queue = self.lock_queue();
            queue.closed = true;
            queue.jobs.drain(..).collect()
        };
        self.wake.notify_all();
        for job in queued {
            for request in job.requests {
                self.count_dropped("shutdown");
                let _ = request.reply.send(Err(ApiError::NotReady));
            }
        }
    }
}

/// Removes a queued job when the future waiting for it is dropped (request
/// timeout, client gone), so a dead job does not hold a place in the queue.
struct AbandonOnDrop<'a> {
    shared: &'a Shared,
    id: Option<u64>,
}

impl Drop for AbandonOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.shared.abandon(id);
        }
    }
}

/// A place in the inference queue, taken before the input is prepared.
///
/// Dropping the ticket gives the place back.
pub struct InferenceTicket {
    slot: Slot,
    deadline: Option<Instant>,
}

impl InferenceTicket {
    /// Queue `prepared` and wait for the worker to run it.
    ///
    /// `timer` records the duration of the model run as the `inference`
    /// stage. If the returned future is dropped while the job is still
    /// queued, the job is removed and never runs.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::Timeout`] if the ticket's deadline passed
    /// before the job started, [`ApiError::NotReady`] if the server is
    /// shutting down, the pipeline's own error if the run failed, and
    /// [`ApiError::Internal`] if the worker lost the job (it panicked).
    pub async fn run(
        self,
        prepared: PreparedData,
        timer: Option<StageTimer>,
    ) -> Result<PipelineOutput, ApiError> {
        let Self { slot, deadline } = self;
        let shared = Arc::clone(&slot.shared);
        let (reply, result) = oneshot::channel();
        let request = Request {
            prepared,
            deadline,
            reply,
            slot,
        };
        let (id, _finished) = shared.push(vec![request], false, timer)?;
        let mut abandon = AbandonOnDrop {
            shared: &shared,
            id: Some(id),
        };
        let result = result.await;
        abandon.id = None;
        result.unwrap_or_else(|_| {
            Err(ApiError::Internal(
                "inference worker dropped the job".to_string(),
            ))
        })
    }
}

/// Handle to the inference worker thread.
///
/// Dropping the handle closes the queue, which ends the thread after the
/// job it is running.
pub(crate) struct InferenceWorker {
    shared: Arc<Shared>,
}

impl InferenceWorker {
    /// Start the worker thread for `pipeline`.
    ///
    /// # Panics
    ///
    /// Panics if the operating system cannot start a thread.
    pub(crate) fn spawn(pipeline: Arc<RwLock<PipelineKind>>, model_id: &str) -> Self {
        let shared = Arc::new(Shared {
            model_id: Arc::from(model_id),
            capacity: AtomicUsize::new(DEFAULT_QUEUE_CAPACITY),
            waiting: AtomicUsize::new(0),
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                closed: false,
                next_id: 0,
            }),
            wake: Condvar::new(),
            #[cfg(test)]
            before_inference: Mutex::new(None),
        });
        let thread_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("inference-worker".to_string())
            .spawn(move || worker_loop(&thread_shared, &pipeline))
            .expect("failed to start the inference worker thread");
        Self { shared }
    }

    /// Set how many requests may wait at one time (at least one).
    pub(crate) fn set_capacity(&self, capacity: usize) {
        self.shared.capacity.store(capacity.max(1), Ordering::SeqCst);
    }

    /// Install the test seam that runs before every job's model run.
    #[cfg(test)]
    pub(crate) fn set_before_inference(&self, hook: impl Fn() + Send + Sync + 'static) {
        *self
            .shared
            .before_inference
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(hook));
    }

    /// Requests admitted and not yet started.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.shared.waiting.load(Ordering::SeqCst)
    }

    /// Take a place in the queue for a request that must be answered by
    /// `deadline`.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::QueueFull`] when as many requests as the
    /// capacity are already waiting.
    pub(crate) fn admit(&self, deadline: Option<Instant>) -> Result<InferenceTicket, ApiError> {
        match self.shared.take_slot(true) {
            Some(slot) => Ok(InferenceTicket { slot, deadline }),
            None => {
                self.shared.count_rejected("queue_full");
                Err(ApiError::QueueFull)
            }
        }
    }

    /// Run a batch collected by the dynamic batcher as one job.
    ///
    /// Every request is answered on its own reply channel. Returns when
    /// the job is finished. The batcher has its own bounded channel and
    /// submits one batch at a time, so a batch is not refused for a full
    /// queue.
    pub(crate) async fn run_batch(&self, batch: Vec<BatchRequest>) {
        let requests: Vec<Request> = batch
            .into_iter()
            .filter_map(|request| {
                Some(Request {
                    prepared: request.prepared,
                    deadline: request.deadline,
                    reply: request.reply,
                    slot: self.shared.take_slot(false)?,
                })
            })
            .collect();
        // A refused batch (shutdown) is answered by dropping its reply
        // channels, like a batch whose run was lost.
        if let Ok((_id, finished)) = self.shared.push(requests, true, None) {
            let _ = finished.await;
        }
    }

    /// Stop accepting jobs and fail the queued ones with
    /// [`ApiError::NotReady`]. The job that is running finishes.
    pub(crate) fn close(&self) {
        self.shared.close();
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.shared.close();
    }
}

/// Body of the worker thread: run queued jobs until the queue is closed.
fn worker_loop(shared: &Arc<Shared>, pipeline: &RwLock<PipelineKind>) {
    while let Some(job) = shared.next_job() {
        // A panic in the model run must not end the thread: the reply
        // channels of the lost job are dropped, which answers its callers.
        if catch_unwind(AssertUnwindSafe(|| run_job(shared, pipeline, job))).is_err() {
            tracing::error!("inference job panicked");
        }
    }
}

/// Run one job on the pipeline and answer its requests.
fn run_job(shared: &Shared, pipeline: &RwLock<PipelineKind>, job: Job) {
    let Job {
        requests,
        batched,
        timer,
        span,
        ..
    } = job;
    let now = Instant::now();
    let (inputs, replies): (Vec<PreparedData>, Vec<Reply>) = requests
        .into_iter()
        .filter_map(|request| request.start(now))
        .unzip();
    if inputs.is_empty() {
        return;
    }

    let _entered = span.enter();
    // Uncontended: requests are prepared without the pipeline, and nothing
    // else takes this lock once the server is up.
    let mut pipeline = pipeline.blocking_write();
    shared.before_inference();

    if batched {
        let results = pipeline.execute_batch(inputs);
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result.map_err(ApiError::from));
        }
    } else {
        for (prepared, reply) in inputs.into_iter().zip(replies) {
            let result = match &timer {
                Some(timer) => timer.time("inference", || pipeline.execute(prepared)),
                None => pipeline.execute(prepared),
            };
            let _ = reply.send(result.map_err(ApiError::from));
        }
    }
}
