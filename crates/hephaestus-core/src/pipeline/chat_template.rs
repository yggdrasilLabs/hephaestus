//! Chat templates: the prompt format an instruction model was trained on.
//!
//! The format ships with the model as a Jinja template -- in
//! `chat_template.jinja`, or under `chat_template` in
//! `tokenizer_config.json`. It is rendered here the way the Hugging Face
//! tokenizer renders it (`apply_chat_template` with
//! `add_generation_prompt=True`): same variables, same whitespace rules,
//! same helper functions. A prompt in the wrong format does not fail; the
//! model just answers worse, so nothing in this module guesses. A model
//! without a template, or with one that cannot be rendered, is rejected
//! when the pipeline is built.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{AutoEscape, Environment, Error as TemplateError, ErrorKind, Value};

use crate::error::CoreError;

/// Template file written by current `transformers` releases.
const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";
/// Tokenizer settings; holds the template in older exports, and the
/// special tokens a template may print.
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
/// Name the template is registered under in the Jinja environment.
const TEMPLATE_NAME: &str = "chat_template";
/// Name of the template to use when a model ships several.
const DEFAULT_TEMPLATE: &str = "default";

/// Special tokens of the tokenizer that a template may refer to by name.
const SPECIAL_TOKEN_KEYS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// A model's chat template, ready to render prompts.
#[derive(Clone)]
pub(super) struct ChatTemplate {
    environment: Environment<'static>,
    /// Special tokens by name (`bos_token`, ...), as template variables.
    special_tokens: serde_json::Map<String, serde_json::Value>,
}

impl ChatTemplate {
    /// Load the chat template of the model in `model_dir`.
    ///
    /// `chat_template.jinja` wins over the `chat_template` entry of
    /// `tokenizer_config.json`, as it does in `transformers`. When the
    /// entry is a list of named templates, the one named `default` is used.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if the model has no chat
    /// template or the template is not valid Jinja, and [`CoreError::Io`] /
    /// [`CoreError::Json`] if a file cannot be read or parsed.
    pub(super) fn load(model_dir: &Path) -> Result<Self, CoreError> {
        let tokenizer_config_path = model_dir.join(TOKENIZER_CONFIG_FILE);
        let tokenizer_config: serde_json::Value = if tokenizer_config_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&tokenizer_config_path)?)?
        } else {
            serde_json::Value::Null
        };

        let template_path = model_dir.join(CHAT_TEMPLATE_FILE);
        let source = if template_path.exists() {
            std::fs::read_to_string(&template_path)?
        } else {
            template_from_tokenizer_config(&tokenizer_config)?
        };

        Self::from_source(source, special_tokens(&tokenizer_config))
    }

    /// Build a template from its Jinja source and the tokenizer's special
    /// tokens.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if the source is not valid
    /// Jinja for this engine.
    pub(super) fn from_source(
        source: String,
        special_tokens: serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, CoreError> {
        let mut environment = Environment::new();
        // The settings `transformers` compiles chat templates with.
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|e| CoreError::ModelValidation(format!("chat template syntax: {e}")))?;
        environment.set_syntax(syntax);
        // A prompt is plain text; nothing in it is HTML to escape.
        environment.set_auto_escape_callback(|_| AutoEscape::None);
        // Templates are written for Python Jinja and call `str` and `dict`
        // methods (`.strip()`, `.startswith()`, `.items()`, ...).
        environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function("raise_exception", raise_exception);
        environment.add_function("strftime_now", strftime_now);
        environment
            .add_template_owned(TEMPLATE_NAME, source)
            .map_err(|e| {
                CoreError::ModelValidation(format!("chat template cannot be parsed: {e:#}"))
            })?;

        Ok(Self {
            environment,
            special_tokens,
        })
    }

    /// Render the prompt for one user message, ending with the marker that
    /// starts the assistant's reply.
    ///
    /// With `system`, the conversation is a system message followed by the
    /// user message; without, the user message alone, and the template
    /// decides whether to add a system message of its own.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidInput`] if the template refuses the
    /// conversation (for example a model that does not accept a system
    /// message) or fails to render.
    pub(super) fn render(&self, system: Option<&str>, user: &str) -> Result<String, CoreError> {
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }
        messages.push(serde_json::json!({ "role": "user", "content": user }));

        // The variables `apply_chat_template` passes: the special tokens,
        // then the conversation. `tools` and `documents` are passed as
        // none, not left undefined, exactly as `transformers` does.
        let mut context = self.special_tokens.clone();
        context.insert("messages".to_string(), serde_json::Value::Array(messages));
        context.insert("tools".to_string(), serde_json::Value::Null);
        context.insert("documents".to_string(), serde_json::Value::Null);
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(true),
        );

        let template = self
            .environment
            .get_template(TEMPLATE_NAME)
            .map_err(|e| CoreError::Inference(format!("chat template is not loaded: {e}")))?;
        template
            .render(Value::from(Serde(&context)))
            .map_err(|e| CoreError::InvalidInput(format!("chat template failed to render: {e:#}")))
    }
}

/// Read the template out of a parsed `tokenizer_config.json`.
fn template_from_tokenizer_config(tokenizer_config: &serde_json::Value) -> Result<String, CoreError> {
    match tokenizer_config.get("chat_template") {
        Some(serde_json::Value::String(source)) => Ok(source.clone()),
        Some(serde_json::Value::Array(named)) => named
            .iter()
            .find(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some(DEFAULT_TEMPLATE))
            .and_then(|entry| entry.get("template"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                CoreError::ModelValidation(format!(
                    "'{TOKENIZER_CONFIG_FILE}' lists chat templates but none is named '{DEFAULT_TEMPLATE}'",
                ))
            }),
        _ => Err(CoreError::ModelValidation(format!(
            "model has no chat template: neither '{CHAT_TEMPLATE_FILE}' nor a 'chat_template' \
             entry in '{TOKENIZER_CONFIG_FILE}' was found; text generation needs the prompt \
             format the model was trained on",
        ))),
    }
}

/// Collect the special tokens a template may print.
///
/// A token is stored either as a string or as an object with a `content`
/// field; both forms yield the plain string. Unset tokens are left out,
/// which makes them undefined in the template, as in `transformers`.
fn special_tokens(tokenizer_config: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    let content = |value: &serde_json::Value| -> Option<String> {
        value
            .as_str()
            .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
            .map(str::to_string)
    };

    let mut tokens = serde_json::Map::new();
    for key in SPECIAL_TOKEN_KEYS {
        if let Some(token) = tokenizer_config.get(key).and_then(content) {
            tokens.insert(key.to_string(), serde_json::Value::String(token));
        }
    }
    if let Some(additional) = tokenizer_config
        .get("additional_special_tokens")
        .and_then(serde_json::Value::as_array)
    {
        let additional: Vec<serde_json::Value> = additional
            .iter()
            .filter_map(content)
            .map(serde_json::Value::String)
            .collect();
        tokens.insert(
            "additional_special_tokens".to_string(),
            serde_json::Value::Array(additional),
        );
    }
    tokens
}

/// `raise_exception(message)`: lets a template refuse a conversation.
fn raise_exception(message: String) -> Result<Value, TemplateError> {
    Err(TemplateError::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)`: the current date and time, formatted.
///
/// Templates use it to print today's date. The time is UTC, and only the
/// directives handled by [`format_utc`] are available.
fn strftime_now(format: String) -> Result<String, TemplateError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    format_utc(&format, seconds)
        .map_err(|directive| {
            TemplateError::new(
                ErrorKind::InvalidOperation,
                format!("strftime_now does not support the directive '%{directive}'"),
            )
        })
}

/// Format a Unix timestamp (UTC) with a subset of `strftime`.
///
/// Supports `%Y %y %m %d %e %b %B %a %A %H %M %S %j %%`. Returns the
/// offending directive character otherwise.
fn format_utc(format: &str, unix_seconds: u64) -> Result<String, char> {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const WEEKDAYS: [&str; 7] = [
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
    ];

    let days = unix_seconds / 86_400;
    let second_of_day = unix_seconds % 86_400;
    // Civil date from a day count (Howard Hinnant's algorithm), with the
    // era starting on 0000-03-01 so leap days fall at the end of a year.
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    let is_leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    const DAYS_BEFORE_MONTH: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let month_slot = usize::try_from(month - 1).unwrap_or(0);
    let ordinal = DAYS_BEFORE_MONTH[month_slot] + day + u64::from(is_leap && month > 2);
    let month_name = MONTHS[month_slot];
    let weekday = WEEKDAYS[usize::try_from(days % 7).unwrap_or(0)];

    let mut out = String::with_capacity(format.len() + 16);
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&year.to_string()),
            Some('y') => out.push_str(&format!("{:02}", year % 100)),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('e') => out.push_str(&format!("{day:2}")),
            Some('b') => out.push_str(&month_name[..3]),
            Some('B') => out.push_str(month_name),
            Some('a') => out.push_str(&weekday[..3]),
            Some('A') => out.push_str(weekday),
            Some('H') => out.push_str(&format!("{:02}", second_of_day / 3_600)),
            Some('M') => out.push_str(&format!("{:02}", second_of_day % 3_600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", second_of_day % 60)),
            Some('j') => out.push_str(&format!("{ordinal:03}")),
            Some('%') => out.push('%'),
            Some(other) => return Err(other),
            None => return Err('%'),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    /// ChatML with a built-in default system message (SmolLM2's template).
    const CHATML_WITH_DEFAULT_SYSTEM: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    fn template(source: &str, tokenizer_config: serde_json::Value) -> ChatTemplate {
        ChatTemplate::from_source(source.to_string(), special_tokens(&tokenizer_config))
            .expect("template should parse")
    }

    #[test]
    fn chatml_template_renders_system_and_user_messages() {
        // Arrange
        let template = template(CHATML_WITH_DEFAULT_SYSTEM, json!({}));

        // Act
        let prompt = template
            .render(Some("Be brief."), "Hello\nthere")
            .expect("render");

        // Assert
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHello\nthere<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_template_adds_its_own_system_message_when_none_is_given() {
        // Arrange
        let template = template(CHATML_WITH_DEFAULT_SYSTEM, json!({}));

        // Act
        let prompt = template.render(None, "Hi").expect("render");

        // Assert
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn user_text_is_not_escaped_or_trimmed() {
        // Arrange
        let template = template(CHATML_WITH_DEFAULT_SYSTEM, json!({}));
        let text = "  <b>a & b</b> {{ not a variable }} \"quoted\"\n";

        // Act
        let prompt = template.render(Some("s"), text).expect("render");

        // Assert
        assert!(prompt.contains(&format!("<|im_start|>user\n{text}<|im_end|>")), "{prompt}");
    }

    #[test]
    fn special_tokens_come_from_the_tokenizer_config() {
        // Arrange -- one token as a string, one as an added-token object.
        let config = json!({
            "bos_token": "<s>",
            "eos_token": { "content": "</s>", "lstrip": false },
            "pad_token": null,
        });
        let template = template(
            "{{ bos_token }}{% for m in messages %}[{{ m.role }}] {{ m.content }}{{ eos_token }}{% endfor %}{% if pad_token is defined %}PAD{% endif %}",
            config,
        );

        // Act
        let prompt = template.render(None, "hi").expect("render");

        // Assert
        assert_eq!(prompt, "<s>[user] hi</s>");
    }

    #[test]
    fn whitespace_control_matches_transformers_settings() {
        // Arrange -- block tags on their own lines, as multi-line templates
        // are written: trim_blocks and lstrip_blocks remove them entirely.
        let source = "{% for message in messages %}\n    {% if message['role'] == 'user' %}\n{{ message['content'] }}\n    {% endif %}\n{% endfor %}\n{%- if add_generation_prompt %}\nA:\n{%- endif %}\n";
        let template = template(source, json!({}));

        // Act
        let prompt = template.render(None, "Q").expect("render");

        // Assert -- what Python Jinja renders with the same settings.
        assert_eq!(prompt, "Q\nA:");
    }

    #[test]
    fn python_string_methods_and_namespaces_work() {
        // Arrange -- constructs common in published templates.
        let source = "{%- set ns = namespace(system='') -%}{%- for m in messages -%}{%- if m['role'] == 'system' -%}{%- set ns.system = m['content'].strip() -%}{%- endif -%}{%- endfor -%}{%- for m in messages[1:] if m.role == 'user' -%}{{ ns.system.upper() }}|{{ m['content'] | trim }}|{{ loop.index0 }}{%- endfor -%}{%- if tools is not none %}TOOLS{% endif -%}";
        let template = template(source, json!({}));

        // Act
        let prompt = template.render(Some("  sys  "), " hi ").expect("render");

        // Assert
        assert_eq!(prompt, "SYS|hi|0");
    }

    #[test]
    fn raise_exception_refuses_the_conversation() {
        // Arrange -- a model that takes no system message.
        let source = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{{ messages[0]['content'] }}";
        let template = template(source, json!({}));

        // Act
        let refused = template.render(Some("s"), "u");
        let accepted = template.render(None, "u");

        // Assert
        let err = refused.expect_err("system message should be refused");
        assert!(matches!(err, CoreError::InvalidInput(_)), "{err:?}");
        assert!(err.to_string().contains("System role not supported"), "{err}");
        assert_eq!(accepted.expect("user-only conversation renders"), "u");
    }

    #[test]
    fn invalid_template_is_rejected_when_loaded() {
        // Arrange / Act
        let result = ChatTemplate::from_source("{% for m in messages %}".to_string(), serde_json::Map::new());

        // Assert
        assert!(matches!(result, Err(CoreError::ModelValidation(_))));
    }

    #[test]
    fn template_file_wins_over_tokenizer_config() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join(CHAT_TEMPLATE_FILE), "file:{{ messages[0].content }}").expect("write");
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_FILE),
            json!({ "chat_template": "config:{{ messages[0].content }}" }).to_string(),
        )
        .expect("write");

        // Act
        let prompt = ChatTemplate::load(dir.path())
            .expect("load")
            .render(None, "x")
            .expect("render");

        // Assert
        assert_eq!(prompt, "file:x");
    }

    #[test]
    fn named_default_template_is_used() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_FILE),
            json!({ "chat_template": [
                { "name": "tool_use", "template": "tools" },
                { "name": "default", "template": "default:{{ messages[0].content }}" },
            ]})
            .to_string(),
        )
        .expect("write");

        // Act
        let prompt = ChatTemplate::load(dir.path())
            .expect("load")
            .render(None, "x")
            .expect("render");

        // Assert
        assert_eq!(prompt, "default:x");
    }

    #[test]
    fn model_without_template_is_rejected() {
        // Arrange -- a tokenizer config without a template, and no config.
        let with_config = tempfile::tempdir().expect("temp dir");
        std::fs::write(with_config.path().join(TOKENIZER_CONFIG_FILE), "{}").expect("write");
        let empty = tempfile::tempdir().expect("temp dir");

        for dir in [with_config.path(), empty.path()] {
            // Act
            let result = ChatTemplate::load(dir);

            // Assert
            let err = result.err().expect("missing template should fail");
            assert!(matches!(err, CoreError::ModelValidation(_)), "{err:?}");
            assert!(err.to_string().contains("no chat template"), "{err}");
        }
    }

    #[test]
    fn utc_formatting_matches_known_dates() {
        // 2024-02-29 13:05:09 UTC, a Thursday in a leap year.
        let leap_day = 1_709_211_909;
        assert_eq!(
            format_utc("%d %b %Y", leap_day).as_deref(),
            Ok("29 Feb 2024")
        );
        assert_eq!(
            format_utc("%A, %B %e %y %H:%M:%S %j %%", leap_day).as_deref(),
            Ok("Thursday, February 29 24 13:05:09 060 %")
        );
        assert_eq!(format_utc("%Y-%m-%d", 0).as_deref(), Ok("1970-01-01"));
        assert_eq!(format_utc("%Y-%m-%d %a", 1_767_225_599).as_deref(), Ok("2025-12-31 Wed"));
        assert_eq!(format_utc("%Z", 0), Err('Z'));
    }

    #[test]
    fn strftime_now_is_available_to_templates() {
        // Arrange
        let template = template("{{ strftime_now('%Y') | length }}", json!({}));

        // Act
        let prompt = template.render(None, "x").expect("render");

        // Assert
        assert_eq!(prompt, "4");
    }

    /// Rendering parity with `transformers` on published chat templates.
    ///
    /// Needs a cases file, so it never runs by default. Point
    /// `HEPHAESTUS_CHAT_TEMPLATE_CASES` at a JSON file of the form
    /// `{"cases": [{"name", "template", "tokenizer_config", "system",
    /// "text", "expected", "error"}]}`, where `expected` is what the
    /// template renders to in `transformers`' own Jinja environment with
    /// `add_generation_prompt=True`, or `error` is set when it refuses the
    /// conversation. A case may add `tokenizer_json` (a path) and
    /// `input_ids`: the rendered prompt must then tokenize to those IDs.
    ///
    /// `cargo test -p hephaestus-core --lib chat_templates_match_reference -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a reference cases file (see doc comment)"]
    fn chat_templates_match_reference_cases() {
        // Arrange
        let cases_path = std::env::var("HEPHAESTUS_CHAT_TEMPLATE_CASES")
            .expect("HEPHAESTUS_CHAT_TEMPLATE_CASES must be set");
        let cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(cases_path).expect("read cases"))
                .expect("parse cases");
        let mut failures = Vec::new();

        for case in cases["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("case name");
            let source = case["template"].as_str().expect("template").to_string();
            let system = case["system"].as_str();
            let text = case["text"].as_str().expect("text");

            // Act
            let rendered = ChatTemplate::from_source(source, special_tokens(&case["tokenizer_config"]))
                .and_then(|template| template.render(system, text));

            // Assert
            let outcome = match (&rendered, case["expected"].as_str()) {
                (Ok(prompt), Some(expected)) if prompt == expected => {
                    match (case["tokenizer_json"].as_str(), case["input_ids"].as_array()) {
                        (Some(path), Some(reference)) => {
                            let tokenizer = tokenizers::Tokenizer::from_file(path).expect("tokenizer");
                            let ids: Vec<u64> = tokenizer
                                .encode(prompt.as_str(), false)
                                .expect("encode")
                                .get_ids()
                                .iter()
                                .map(|&id| u64::from(id))
                                .collect();
                            let reference: Vec<u64> =
                                reference.iter().map(|id| id.as_u64().expect("id")).collect();
                            if ids == reference {
                                format!("ok ({} chars, {} token IDs equal)", prompt.len(), ids.len())
                            } else {
                                failures.push(name.to_string());
                                "TOKEN IDS DIFFER".to_string()
                            }
                        }
                        _ => format!("ok ({} chars)", prompt.len()),
                    }
                }
                (Err(error), None) => format!("ok (both refuse: {error})"),
                (Ok(prompt), Some(expected)) => {
                    failures.push(name.to_string());
                    format!("DIFFERS\n  expected: {expected:?}\n  rendered: {prompt:?}")
                }
                (Ok(_), None) => {
                    failures.push(name.to_string());
                    "DIFFERS: reference refuses, this engine renders".to_string()
                }
                (Err(error), Some(_)) => {
                    failures.push(name.to_string());
                    format!("FAILED: {error}")
                }
            };
            println!("{name}: {outcome}");
        }
        assert!(failures.is_empty(), "templates that differ: {failures:?}");
    }
}
