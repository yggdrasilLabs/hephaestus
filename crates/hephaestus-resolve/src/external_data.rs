//! ONNX external data: weight files that live next to a graph file.
//!
//! A graph larger than 2 GB (and many smaller exports) keeps its weights
//! outside the `.onnx` file. Each such tensor names the file that holds its
//! bytes, relative to the directory of the graph. The file has no fixed
//! name: Optimum writes `model.onnx_data`, the PyTorch exporter
//! `model.onnx.data`, transformers.js splits into `model.onnx_data_1`,
//! `_2`, ..., and older exports write one file per tensor. The only
//! reliable source is the graph itself, so this module reads the file
//! names out of the graph instead of guessing them.
//!
//! A graph without external data yields an empty list, and so does a file
//! that is not a readable ONNX graph. Callers therefore do nothing extra
//! for models that keep their weights inside the graph file.

use std::collections::BTreeSet;
use std::io::{self, BufReader, Read, Seek};
use std::path::{Component, Path, PathBuf};

/// Nesting depth at which a graph is rejected as malformed.
///
/// Real graphs nest a handful of levels (model, graph, node, attribute,
/// subgraph, ...); the limit only stops a crafted file from exhausting
/// the stack.
const MAX_DEPTH: usize = 64;

/// Longest accepted `external_data` entry (a key/value pair), in bytes.
const MAX_ENTRY_LEN: u64 = 16 * 1024;

/// Protobuf message types of the ONNX schema that can lead to a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Message {
    Model,
    Graph,
    Node,
    Attribute,
    Function,
    SparseTensor,
    Tensor,
}

impl Message {
    /// The message type of a length-delimited field, when it can hold a
    /// tensor. Field numbers are those of `onnx.proto`.
    fn child(self, field: u64) -> Option<Self> {
        match (self, field) {
            // ModelProto: graph = 7, functions = 25.
            (Self::Model, 7) => Some(Self::Graph),
            (Self::Model, 25) => Some(Self::Function),
            // GraphProto: node = 1, initializer = 5, sparse_initializer = 15.
            (Self::Graph, 1) => Some(Self::Node),
            (Self::Graph, 5) => Some(Self::Tensor),
            (Self::Graph, 15) => Some(Self::SparseTensor),
            // FunctionProto: node = 7.
            (Self::Function, 7) => Some(Self::Node),
            // NodeProto: attribute = 5.
            (Self::Node, 5) => Some(Self::Attribute),
            // AttributeProto: t = 5, g = 6, tensors = 10, graphs = 11,
            // sparse_tensor = 22, sparse_tensors = 23.
            (Self::Attribute, 5 | 10) => Some(Self::Tensor),
            (Self::Attribute, 6 | 11) => Some(Self::Graph),
            (Self::Attribute, 22 | 23) => Some(Self::SparseTensor),
            // SparseTensorProto: values = 1, indices = 2.
            (Self::SparseTensor, 1 | 2) => Some(Self::Tensor),
            _ => None,
        }
    }
}

/// `TensorProto.external_data`: repeated key/value entries.
const TENSOR_EXTERNAL_DATA_FIELD: u64 = 13;
/// Key of the entry that names the file holding a tensor's bytes.
const LOCATION_KEY: &[u8] = b"location";

fn malformed(reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("not a readable ONNX graph: {reason}"),
    )
}

/// Reads protobuf wire format from a seekable source, skipping payloads
/// (tensor bytes above all) without reading them.
struct WireReader<R> {
    source: BufReader<R>,
    position: u64,
}

impl<R: Read + Seek> WireReader<R> {
    fn byte(&mut self) -> io::Result<u8> {
        let mut byte = [0_u8; 1];
        self.source.read_exact(&mut byte)?;
        self.position += 1;
        Ok(byte[0])
    }

    fn varint(&mut self) -> io::Result<u64> {
        let mut value = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(malformed("varint is longer than 10 bytes"))
    }

    fn skip(&mut self, len: u64) -> io::Result<()> {
        let offset = i64::try_from(len).map_err(|_| malformed("field length overflows"))?;
        self.source.seek_relative(offset)?;
        self.position += len;
        Ok(())
    }

    fn bytes(&mut self, len: u64) -> io::Result<Vec<u8>> {
        let len = usize::try_from(len).map_err(|_| malformed("field length overflows"))?;
        let mut data = vec![0_u8; len];
        self.source.read_exact(&mut data)?;
        self.position += len as u64;
        Ok(data)
    }

    /// Walk one message that ends at byte `end`, collecting the external
    /// data locations of every tensor below it.
    fn message(
        &mut self,
        kind: Message,
        end: u64,
        depth: usize,
        locations: &mut BTreeSet<String>,
    ) -> io::Result<()> {
        if depth > MAX_DEPTH {
            return Err(malformed("messages are nested too deeply"));
        }
        while self.position < end {
            let tag = self.varint()?;
            let field = tag >> 3;
            match tag & 0x7 {
                0 => {
                    self.varint()?;
                }
                1 => self.skip(8)?,
                5 => self.skip(4)?,
                2 => {
                    let len = self.varint()?;
                    let field_end = self
                        .position
                        .checked_add(len)
                        .filter(|field_end| *field_end <= end)
                        .ok_or_else(|| malformed("field runs past the end of its message"))?;
                    if kind == Message::Tensor && field == TENSOR_EXTERNAL_DATA_FIELD {
                        if len > MAX_ENTRY_LEN {
                            return Err(malformed("external data entry is too long"));
                        }
                        if let Some(location) = self.location_entry(field_end)? {
                            locations.insert(location);
                        }
                    } else if let Some(child) = kind.child(field) {
                        self.message(child, field_end, depth + 1, locations)?;
                    } else {
                        self.skip(len)?;
                    }
                }
                _ => return Err(malformed("unsupported wire type")),
            }
        }
        if self.position == end {
            Ok(())
        } else {
            Err(malformed("field runs past the end of its message"))
        }
    }

    /// Read one `StringStringEntryProto` (key = 1, value = 2) that ends at
    /// `end`; return its value when the key is `location`.
    fn location_entry(&mut self, end: u64) -> io::Result<Option<String>> {
        let mut key = Vec::new();
        let mut value = Vec::new();
        while self.position < end {
            let tag = self.varint()?;
            if tag & 0x7 != 2 {
                return Err(malformed("external data entry holds a non-string field"));
            }
            let len = self.varint()?;
            if self
                .position
                .checked_add(len)
                .is_none_or(|field_end| field_end > end)
            {
                return Err(malformed("field runs past the end of its message"));
            }
            match tag >> 3 {
                1 => key = self.bytes(len)?,
                2 => value = self.bytes(len)?,
                _ => self.skip(len)?,
            }
        }
        if key != LOCATION_KEY {
            return Ok(None);
        }
        String::from_utf8(value)
            .map(Some)
            .map_err(|_| malformed("external data location is not UTF-8"))
    }
}

/// Check that an external data location stays inside the graph's directory.
///
/// ONNX Runtime refuses absolute locations and locations that climb out of
/// the model directory; fetching such a path would also let a graph name
/// files outside its own model.
fn is_safe_location(location: &str) -> bool {
    !location.is_empty()
        && !location.contains('\\')
        && Path::new(location)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Collect the external data locations of a graph, exactly as written:
/// relative to the directory that holds the graph, sorted and without
/// duplicates. An empty list means the graph carries all of its weights.
fn scan<R: Read + Seek>(source: R, len: u64) -> io::Result<Vec<String>> {
    let mut reader = WireReader {
        source: BufReader::new(source),
        position: 0,
    };
    let mut found = BTreeSet::new();
    reader.message(Message::Model, len, 0, &mut found)?;
    Ok(found.into_iter().collect())
}

/// Reject a location that leaves the directory of its graph.
fn require_safe(found: Vec<String>) -> io::Result<Vec<String>> {
    match found.iter().find(|location| !is_safe_location(location)) {
        Some(location) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("external data location '{location}' leaves the model directory"),
        )),
        None => Ok(found),
    }
}

/// List the external data files of the ONNX graph at `path`.
///
/// A file that is not a readable ONNX graph has no external data as far as
/// the resolver is concerned: it is handled exactly as it was before
/// external data was supported, and loading it reports the real problem.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, or if the graph names a
/// location outside its own directory.
pub(crate) fn locations_of_file(path: &Path) -> io::Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    match scan(file, len) {
        Ok(found) => require_safe(found),
        Err(error) => {
            tracing::debug!(
                path = %path.display(),
                %error,
                "could not read ONNX graph structure; assuming no external data"
            );
            Ok(Vec::new())
        }
    }
}

/// [`locations_of_file`] off the async executor.
///
/// # Errors
///
/// As [`locations_of_file`].
pub(crate) async fn locations_of_graph(path: PathBuf) -> io::Result<Vec<String>> {
    tokio::task::spawn_blocking(move || locations_of_file(&path))
        .await
        .map_err(io::Error::other)?
}

/// [`external_files`] off the async executor.
///
/// # Errors
///
/// As [`external_files`].
pub(crate) async fn external_files_of_dir(model_dir: PathBuf) -> io::Result<Vec<ExternalFile>> {
    tokio::task::spawn_blocking(move || external_files(&model_dir))
        .await
        .map_err(io::Error::other)?
}

/// An external data file a graph in a model directory needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalFile {
    /// Path of the data file, relative to the model directory.
    pub(crate) relative_path: PathBuf,
    /// Whether the file exists in the model directory.
    pub(crate) present: bool,
}

/// List the external data files needed by every `.onnx` graph under
/// `model_dir` (searched recursively, symbolic links followed).
///
/// # Errors
///
/// Returns an error if a directory or graph cannot be read, or if a graph
/// names a location outside its own directory.
pub(crate) fn external_files(model_dir: &Path) -> io::Result<Vec<ExternalFile>> {
    fn walk(base: &Path, dir: &Path, depth: usize, out: &mut Vec<ExternalFile>) -> io::Result<()> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(base, &path, depth + 1, out)?;
            } else if path.extension().is_some_and(|ext| ext == "onnx") {
                for location in locations_of_file(&path)? {
                    let data_path = dir.join(location);
                    let relative_path = data_path
                        .strip_prefix(base)
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|_| data_path.clone());
                    if !out.iter().any(|file| file.relative_path == relative_path) {
                        out.push(ExternalFile {
                            relative_path,
                            present: data_path.is_file(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    walk(model_dir, model_dir, 0, &mut files)?;
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(files)
}

/// Rebuild `source` under `target` out of regular files.
///
/// The Hugging Face cache stores every file once under `blobs/` and
/// exposes it through a symbolic link in the snapshot directory. ONNX
/// Runtime resolves an external data path to its real location and then
/// refuses it, because that location is outside the directory of the
/// graph. Hard-linking each blob into a plain directory gives ONNX Runtime
/// a graph and its data side by side without storing the bytes twice; a
/// copy is made only where a hard link is not possible (another
/// filesystem).
///
/// The directory is built under a temporary name and renamed into place,
/// replacing an earlier one, so a reader never sees a partial tree.
///
/// # Errors
///
/// Returns an error if a file cannot be linked or copied, or the target
/// cannot be put in place.
pub(crate) fn materialize(source: &Path, target: &Path) -> io::Result<()> {
    fn mirror(from: &Path, to: &Path, depth: usize) -> io::Result<()> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        std::fs::create_dir_all(to)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            let path = entry.path();
            let destination = to.join(entry.file_name());
            if path.is_dir() {
                mirror(&path, &destination, depth + 1)?;
            } else {
                // Link the file the entry points at, not the link itself.
                let real = std::fs::canonicalize(&path)?;
                if std::fs::hard_link(&real, &destination).is_err() {
                    std::fs::copy(&real, &destination)?;
                }
            }
        }
        Ok(())
    }

    let parent = target
        .parent()
        .ok_or_else(|| io::Error::other("materialized model directory has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let staging = tempfile::TempDir::new_in(parent)?;
    mirror(source, staging.path(), 0)?;
    if target.exists() {
        std::fs::remove_dir_all(target)?;
    }
    std::fs::rename(staging.path(), target)?;
    // The directory now lives under its final name; nothing left to remove.
    let _ = staging.keep();
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Hand-encoded ONNX graphs for tests: just enough protobuf to carry
    //! tensors with and without external data.

    fn varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    pub(crate) fn field(number: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(number << 3 | 2, &mut out);
        varint(payload.len() as u64, &mut out);
        out.extend_from_slice(payload);
        out
    }

    fn varint_field(number: u64, value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        varint(number << 3, &mut out);
        varint(value, &mut out);
        out
    }

    fn entry(key: &str, value: &str) -> Vec<u8> {
        [field(1, key.as_bytes()), field(2, value.as_bytes())].concat()
    }

    /// A `TensorProto` whose bytes live in `location`.
    pub(crate) fn external_tensor(name: &str, location: &str) -> Vec<u8> {
        [
            varint_field(1, 4), // dims
            varint_field(2, 1), // data_type = FLOAT
            field(8, name.as_bytes()),
            field(13, &entry("location", location)),
            field(13, &entry("offset", "0")),
            field(13, &entry("length", "16")),
            varint_field(14, 1), // data_location = EXTERNAL
        ]
        .concat()
    }

    /// A `TensorProto` that carries its own bytes.
    pub(crate) fn inline_tensor(name: &str) -> Vec<u8> {
        [
            varint_field(1, 4),
            varint_field(2, 1),
            field(8, name.as_bytes()),
            field(9, &[0_u8; 16]), // raw_data
        ]
        .concat()
    }

    /// A `ModelProto` whose graph holds the given initializers.
    pub(crate) fn model(initializers: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = field(2, b"graph");
        for tensor in initializers {
            graph.extend(field(5, tensor));
        }
        [varint_field(1, 8), field(7, &graph)].concat()
    }

    /// A model whose only weight lives in `location`.
    pub(crate) fn model_with_external_data(location: &str) -> Vec<u8> {
        model(&[inline_tensor("bias"), external_tensor("weight", location)])
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{external_tensor, field, inline_tensor, model};
    use super::*;

    use std::io::Cursor;

    fn locations_in(bytes: &[u8]) -> io::Result<Vec<String>> {
        scan(Cursor::new(bytes), bytes.len() as u64).and_then(require_safe)
    }

    #[test]
    fn graph_without_external_data_has_no_locations() {
        // Arrange
        let graph = model(&[inline_tensor("weight"), inline_tensor("bias")]);

        // Act
        let found = locations_in(&graph).expect("graph should parse");

        // Assert
        assert!(found.is_empty());
    }

    #[test]
    fn external_data_location_is_reported_once() {
        // Arrange -- two tensors in the same data file, as Optimum writes.
        let graph = model(&[
            external_tensor("a", "model.onnx_data"),
            inline_tensor("b"),
            external_tensor("c", "model.onnx_data"),
        ]);

        // Act
        let found = locations_in(&graph).expect("graph should parse");

        // Assert
        assert_eq!(found, vec!["model.onnx_data"]);
    }

    #[test]
    fn every_distinct_location_is_reported() {
        // Arrange -- chunked data files and a PyTorch-style name.
        let graph = model(&[
            external_tensor("a", "model.onnx_data_1"),
            external_tensor("b", "model.onnx_data"),
            external_tensor("c", "model.onnx.data"),
        ]);

        // Act
        let found = locations_in(&graph).expect("graph should parse");

        // Assert
        assert_eq!(
            found,
            vec!["model.onnx.data", "model.onnx_data", "model.onnx_data_1"]
        );
    }

    #[test]
    fn tensors_inside_node_attributes_and_subgraphs_are_found() {
        // Arrange -- node -> attribute -> subgraph -> initializer, the shape
        // of a merged decoder's `If` branches, plus a constant tensor.
        let subgraph = field(5, &external_tensor("w", "branch.data"));
        let graph_attribute = field(6, &subgraph);
        let tensor_attribute = field(5, &external_tensor("c", "constant.data"));
        let node = [field(5, &graph_attribute), field(5, &tensor_attribute)].concat();
        let graph = field(1, &node);
        let bytes = field(7, &graph);

        // Act
        let found = locations_in(&bytes).expect("graph should parse");

        // Assert
        assert_eq!(found, vec!["branch.data", "constant.data"]);
    }

    #[test]
    fn location_outside_the_model_directory_is_rejected() {
        for location in ["../secret", "/etc/passwd", "a/../../b", ""] {
            // Arrange
            let graph = model(&[external_tensor("w", location)]);

            // Act
            let result = locations_in(&graph);

            // Assert
            assert!(result.is_err(), "location '{location}' should be rejected");
        }
    }

    #[test]
    fn subdirectory_location_is_accepted() {
        // Arrange
        let graph = model(&[external_tensor("w", "weights/part0.bin")]);

        // Act
        let found = locations_in(&graph).expect("graph should parse");

        // Assert
        assert_eq!(found, vec!["weights/part0.bin"]);
    }

    #[test]
    fn bytes_that_are_not_a_graph_are_an_error() {
        for bytes in [&b"data"[..], b"fake model", b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"] {
            assert!(locations_in(bytes).is_err(), "{bytes:?} should not parse");
        }
    }

    #[test]
    fn truncated_graph_is_an_error() {
        // Arrange
        let graph = model(&[external_tensor("w", "model.onnx_data")]);

        // Act
        let result = locations_in(&graph[..graph.len() - 3]);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn file_that_is_not_a_graph_has_no_external_data() {
        // Arrange -- what the resolver's own tests store as a "model".
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        std::fs::write(&path, b"fake model").unwrap();

        // Act
        let found = locations_of_file(&path).expect("unreadable graph is not an error");

        // Assert
        assert!(found.is_empty());
    }

    #[test]
    fn graph_written_by_the_onnx_library_reports_its_data_file() {
        // Arrange -- `tests/fixtures/external_data/model.onnx`, saved by
        // `onnx.save_model(..., save_as_external_data=True)`.
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/external_data");

        // Act
        let found = locations_of_file(&fixtures.join("model.onnx")).expect("fixture should parse");
        let files = external_files(&fixtures).expect("scan should succeed");

        // Assert
        assert_eq!(found, vec!["model.onnx_data"]);
        assert_eq!(
            files,
            vec![ExternalFile {
                relative_path: PathBuf::from("model.onnx_data"),
                present: true,
            }]
        );
    }

    #[test]
    fn graphs_with_inline_weights_report_no_data_file() {
        // Arrange -- the seq2seq fixtures of the core crate: real graphs,
        // If-subgraphs included, all weights inline.
        let fixtures =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../hephaestus-core/tests/fixtures/seq2seq");

        for name in [
            "model.onnx",
            "encoder_model.onnx",
            "decoder_model.onnx",
            "decoder_model_merged.onnx",
        ] {
            // Act
            let bytes = std::fs::read(fixtures.join(name)).expect("fixture should exist");
            let found = locations_in(&bytes);

            // Assert -- parsed as a graph, not merely tolerated.
            assert_eq!(found.expect("fixture should parse"), Vec::<String>::new(), "{name}");
        }
    }

    #[test]
    fn external_files_reports_presence_per_graph_directory() {
        // Arrange -- one graph with its data, one without.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
        std::fs::write(
            dir.path().join("onnx/encoder_model.onnx"),
            model(&[external_tensor("w", "encoder_model.onnx_data")]),
        )
        .unwrap();
        std::fs::write(dir.path().join("onnx/encoder_model.onnx_data"), [0_u8; 16]).unwrap();
        std::fs::write(
            dir.path().join("onnx/decoder_model.onnx"),
            model(&[external_tensor("w", "decoder_model.onnx_data")]),
        )
        .unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();

        // Act
        let files = external_files(dir.path()).expect("scan should succeed");

        // Assert
        assert_eq!(
            files,
            vec![
                ExternalFile {
                    relative_path: PathBuf::from("onnx/decoder_model.onnx_data"),
                    present: false,
                },
                ExternalFile {
                    relative_path: PathBuf::from("onnx/encoder_model.onnx_data"),
                    present: true,
                },
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn materialize_replaces_cache_links_with_regular_files() {
        // Arrange -- the Hugging Face cache layout: blobs plus a snapshot
        // of relative symbolic links.
        let cache = tempfile::tempdir().unwrap();
        let blobs = cache.path().join("blobs");
        let snapshot = cache.path().join("snapshots/abc");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(snapshot.join("onnx")).unwrap();
        std::fs::write(blobs.join("111"), b"graph").unwrap();
        std::fs::write(blobs.join("222"), b"weights").unwrap();
        std::fs::write(blobs.join("333"), b"{}").unwrap();
        std::os::unix::fs::symlink("../../../blobs/111", snapshot.join("onnx/model.onnx")).unwrap();
        std::os::unix::fs::symlink("../../../blobs/222", snapshot.join("onnx/model.onnx_data"))
            .unwrap();
        std::os::unix::fs::symlink("../../blobs/333", snapshot.join("config.json")).unwrap();
        let target = cache.path().join("materialized/org/model");
        // A leftover from an earlier revision must not survive.
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("stale.txt"), b"old").unwrap();

        // Act
        materialize(&snapshot, &target).expect("materialize should succeed");

        // Assert
        for (name, content) in [
            ("onnx/model.onnx", &b"graph"[..]),
            ("onnx/model.onnx_data", b"weights"),
            ("config.json", b"{}"),
        ] {
            let path = target.join(name);
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            assert!(metadata.file_type().is_file(), "{name} must be a regular file");
            assert_eq!(std::fs::read(&path).unwrap(), content);
        }
        assert!(!target.join("stale.txt").exists());
    }
}
