//! Bridges Rig's completion request/response model to the local `core-llm` engine: render a
//! [`CompletionRequest`] (preamble + tool definitions + chat history) into the engine's
//! [`ChatMessage`] list, and parse the model's raw text back into Rig's [`AssistantContent`].
//! The local model has no native function-calling API, so tools are described in the system prompt
//! and the model emits `<tool_call>{"name":..,"arguments":..}</tool_call>` (Qwen3/Hermes style).
//! Small quantized models often drop the wrapper, so [`parse_choice`] also accepts a bare
//! top-level JSON object (optionally under a `tool_call` key, optionally fenced). Anything that
//! still fails to parse is returned as text, but [`looks_like_unparsed_tool_call`] flags text that
//! resembles an attempted call so callers do not show it to the user as an answer.

use core_llm::{ChatMessage, Role};
use rig::completion::message::{AssistantContent, Message, ToolCall, ToolFunction, UserContent};
use rig::completion::CompletionRequest;
use rig::OneOrMany;

/// The opening/closing markers we ask the model to wrap a tool call in.
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// If `text` is a single fenced code block (` ``` ` or ` ```json `) and nothing else, return its
/// inner content; `None` otherwise. Lets a model fence its tool call; a fenced code sample only
/// counts as a call if it has a `name` field.
fn unfence(text: &str) -> Option<&str> {
    let body = text.strip_prefix("```")?;
    let body = body.strip_prefix("json").unwrap_or(body);
    let body = body.trim_start_matches(['\n', '\r']);
    body.strip_suffix("```").map(str::trim)
}

/// What a JSON-object-shaped string turns out to be.
pub(crate) enum JsonShape {
    /// A tool call: `{"name": ..., "arguments": ...}` or `{"tool_call": {"name": ..., ...}}`.
    ToolCall(String, serde_json::Value),
    /// Valid JSON, but not shaped like a call (no `name`/`tool_call` key).
    Other,
    /// Not valid JSON at all, despite the brace shape.
    Malformed,
}

/// Classify a string expected to be a JSON object (already brace-balanced, e.g. from the streaming
/// gate) as a tool call, unrelated JSON, or malformed. `pub(crate)` so `model::StreamGate` can
/// reuse it while a bare object is still arriving.
pub(crate) fn classify_json_object(text: &str) -> JsonShape {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(val) => match val.as_object() {
            Some(obj) => {
                let call = obj
                    .get("tool_call")
                    .and_then(|v| v.as_object())
                    .unwrap_or(obj);
                match call.get("name").and_then(|v| v.as_str()) {
                    Some(name) => {
                        let args = call
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        JsonShape::ToolCall(name.to_string(), args)
                    }
                    None => JsonShape::Other,
                }
            }
            None => JsonShape::Other,
        },
        Err(_) => JsonShape::Malformed,
    }
}

/// A bare (unwrapped) tool call: the entire trimmed message, or its single fenced code block, is
/// one JSON object naming a tool. `serde_json::from_str` rejects trailing content, so a message
/// that only contains JSON inside a sentence never matches.
fn bare_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
    let text = text.trim();
    let candidate = unfence(text).unwrap_or(text).trim();
    match classify_json_object(candidate) {
        JsonShape::ToolCall(name, args) => Some((name, args)),
        JsonShape::Other | JsonShape::Malformed => None,
    }
}

/// True when `text` did not parse into a tool call but looks like an attempt: a stray
/// `<tool_call>` tag (unclosed, or invalid JSON inside), or a message starting with `{` that is
/// call-shaped without parsing cleanly or fails to parse as JSON. Such text must surface as a
/// failure, not be shown as the answer. Ordinary prose rarely opens with a bare `{`.
pub fn looks_like_unparsed_tool_call(text: &str) -> bool {
    let t = text.trim();
    if t.contains(TOOL_OPEN) {
        return true;
    }
    let candidate = unfence(t).unwrap_or(t).trim();
    if !candidate.starts_with('{') {
        return false;
    }
    matches!(
        classify_json_object(candidate),
        JsonShape::ToolCall(..) | JsonShape::Malformed
    )
}

/// Build the tool-calling instruction block appended to the system prompt: each tool with its JSON
/// argument schema and the wire format to emit. Empty when there are no tools.
fn tools_instruction(req: &CompletionRequest) -> String {
    if req.tools.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "\n\nYou can call tools to look up real data about this board. Available tools:\n",
    );
    for t in &req.tools {
        s.push_str(&format!(
            "- {}: {}\n  arguments JSON schema: {}\n",
            t.name, t.description, t.parameters
        ));
    }
    s.push_str(
        "\nTo call a tool, reply with ONLY a single line of the exact form:\n\
         <tool_call>{\"name\": \"<tool name>\", \"arguments\": { ... }}</tool_call>\n\
         Call a tool when you need data you do not already have. After you receive the tool result (as a \
         user message), use it to answer. When you have enough information, reply with the final answer in \
         plain prose and do NOT emit a tool call. Ground every claim in tool results; never invent data.",
    );
    s
}

/// Flatten a Rig [`Message`] into the text the local chat template will see. Tool calls/results
/// are rendered back into the same `<tool_call>` / tool-result text shape.
fn message_text(msg: &Message) -> (Role, String) {
    match msg {
        Message::System { content } => (Role::System, content.clone()),
        Message::User { content } => {
            let mut parts = Vec::new();
            for c in content.iter() {
                match c {
                    UserContent::Text(t) => parts.push(t.text.clone()),
                    UserContent::ToolResult(tr) => {
                        let body = tr
                            .content
                            .iter()
                            .map(|rc| match rc {
                                rig::completion::message::ToolResultContent::Text(t) => {
                                    t.text.clone()
                                }
                                _ => String::new(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        parts.push(format!("Tool result ({}): {body}", tr.id));
                    }
                    _ => {}
                }
            }
            (Role::User, parts.join("\n"))
        }
        Message::Assistant { content, .. } => {
            let mut parts = Vec::new();
            for c in content.iter() {
                match c {
                    AssistantContent::Text(t) => parts.push(t.text.clone()),
                    AssistantContent::ToolCall(tc) => parts.push(format!(
                        "{TOOL_OPEN}{{\"name\": \"{}\", \"arguments\": {}}}{TOOL_CLOSE}",
                        tc.function.name, tc.function.arguments
                    )),
                    _ => {}
                }
            }
            (Role::Assistant, parts.join("\n"))
        }
    }
}

/// Render a Rig completion request into the engine's message list: one system message (preamble +
/// tool instructions, merged with any system turns), then the user/assistant history in order.
pub fn render_messages(req: &CompletionRequest) -> Vec<ChatMessage> {
    let mut system = req.preamble.clone().unwrap_or_default();
    let mut rest = Vec::new();
    for msg in req.chat_history.iter() {
        let (role, text) = message_text(msg);
        if role == Role::System {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&text);
        } else {
            rest.push(ChatMessage {
                role,
                content: text,
            });
        }
    }
    system.push_str(&tools_instruction(req));

    let mut out = Vec::with_capacity(rest.len() + 1);
    if !system.trim().is_empty() {
        out.push(ChatMessage::system(system));
    }
    out.extend(rest);
    out
}

/// Parse the model's raw answer into Rig assistant content: `<tool_call>{...}</tool_call>` blocks
/// become [`AssistantContent::ToolCall`]s; failing that, a bare top-level JSON object (see
/// [`bare_tool_call`]) becomes a single tool call; otherwise the whole text is one
/// [`AssistantContent::Text`]. A malformed attempt is left as text; callers that show text to a
/// user must also check [`looks_like_unparsed_tool_call`].
pub fn parse_choice(text: &str) -> OneOrMany<AssistantContent> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(TOOL_OPEN) {
        let after = &rest[start + TOOL_OPEN.len()..];
        let Some(end) = after.find(TOOL_CLOSE) else {
            break;
        };
        let json = after[..end].trim();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json) {
            if let Some(name) = val.get("name").and_then(|v| v.as_str()) {
                let args = val
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let id = format!("call-{}", calls.len());
                calls.push(AssistantContent::ToolCall(ToolCall::new(
                    id,
                    ToolFunction::new(name.to_string(), args),
                )));
            }
        }
        rest = &after[end + TOOL_CLOSE.len()..];
    }
    if calls.is_empty() {
        if let Some((name, args)) = bare_tool_call(text) {
            let call = AssistantContent::ToolCall(ToolCall::new(
                "call-0".to_string(),
                ToolFunction::new(name, args),
            ));
            return OneOrMany::one(call);
        }
    }
    if calls.is_empty() {
        let t = text.trim();
        return OneOrMany::one(AssistantContent::text(if t.is_empty() {
            "(no answer)"
        } else {
            t
        }));
    }
    OneOrMany::many(calls).expect("calls is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pull the tool calls out of a parsed choice as (name, args) pairs.
    fn tool_calls(text: &str) -> Vec<(String, serde_json::Value)> {
        parse_choice(text)
            .into_iter()
            .filter_map(|c| match c {
                AssistantContent::ToolCall(tc) => Some((tc.function.name, tc.function.arguments)),
                _ => None,
            })
            .collect()
    }

    /// The single text answer of a parsed choice, if it is plain text.
    fn text_of(text: &str) -> Option<String> {
        parse_choice(text).into_iter().find_map(|c| match c {
            AssistantContent::Text(t) => Some(t.text),
            _ => None,
        })
    }

    #[test]
    fn plain_text_is_one_text_block() {
        assert_eq!(
            text_of("8 PRs await review.").as_deref(),
            Some("8 PRs await review.")
        );
        assert!(tool_calls("8 PRs await review.").is_empty());
    }

    #[test]
    fn extracts_a_tool_call() {
        let out =
            tool_calls("<tool_call>{\"name\": \"board_attention\", \"arguments\": {}}</tool_call>");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "board_attention");
        assert_eq!(out[0].1, serde_json::json!({}));
    }

    #[test]
    fn extracts_tool_call_with_args_ignoring_surrounding_text() {
        let out = tool_calls(
            "let me look\n<tool_call>{\"name\": \"search_code\", \"arguments\": {\"query\": \"auth\", \"k\": 3}}</tool_call>",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "search_code");
        assert_eq!(out[0].1, serde_json::json!({ "query": "auth", "k": 3 }));
    }

    #[test]
    fn extracts_multiple_tool_calls() {
        let out = tool_calls(
            "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>",
        );
        assert_eq!(
            out.iter().map(|c| c.0.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn malformed_tool_call_falls_back_to_text() {
        // Not valid JSON inside the tags -> no tool call; treated as a plain text answer.
        let text = "<tool_call>{not json}</tool_call>";
        assert!(tool_calls(text).is_empty());
        assert!(text_of(text).is_some());
    }

    #[test]
    fn empty_output_is_not_empty_text() {
        // OneOrMany cannot be empty; blank output becomes a placeholder rather than panicking.
        assert_eq!(text_of("   ").as_deref(), Some("(no answer)"));
    }

    #[test]
    fn bare_json_object_is_a_tool_call() {
        let out = tool_calls(r#"{"name": "board_attention", "arguments": {}}"#);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "board_attention");
        assert_eq!(out[0].1, serde_json::json!({}));
    }

    #[test]
    fn bare_json_object_with_surrounding_whitespace_is_a_tool_call() {
        let out = tool_calls("  \n{\"name\": \"whats_stuck\", \"arguments\": {\"limit\": 5}}\n  ");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "whats_stuck");
        assert_eq!(out[0].1, serde_json::json!({ "limit": 5 }));
    }

    #[test]
    fn fenced_json_block_is_a_tool_call() {
        let out = tool_calls("```json\n{\"name\": \"failing_ci\", \"arguments\": {}}\n```");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "failing_ci");
    }

    #[test]
    fn fenced_block_without_json_tag_is_a_tool_call() {
        let out = tool_calls("```\n{\"name\": \"list_teams\", \"arguments\": {}}\n```");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "list_teams");
    }

    #[test]
    fn tool_call_key_wrapped_object_is_a_tool_call() {
        let out =
            tool_calls(r#"{"tool_call": {"name": "search_code", "arguments": {"query": "auth"}}}"#);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "search_code");
        assert_eq!(out[0].1, serde_json::json!({ "query": "auth" }));
    }

    #[test]
    fn prose_with_incidental_json_is_not_a_tool_call() {
        // The object is not the whole message, just something the answer mentions -> plain text.
        let text = r#"Sure - here's an example config: {"name": "x"} should work in your file."#;
        assert!(tool_calls(text).is_empty());
        assert_eq!(text_of(text).as_deref(), Some(text));
    }

    #[test]
    fn json_without_a_name_field_is_plain_text_not_a_failed_call() {
        // Valid JSON, but not call-shaped: a legitimate structured answer, not a tool call.
        let text = r#"{"open_prs": 3, "failing_ci": 1}"#;
        assert!(tool_calls(text).is_empty());
        assert_eq!(text_of(text).as_deref(), Some(text));
        assert!(!looks_like_unparsed_tool_call(text));
    }

    #[test]
    fn unclosed_tool_call_tag_is_flagged_as_an_unparsed_attempt() {
        // No closing tag: parse_choice falls back to text, but the text is flagged so a caller
        // does not show it to the user as an answer.
        let text = "<tool_call>{\"name\": \"board_attention\", \"arguments\": {}}";
        assert!(tool_calls(text).is_empty());
        assert!(text_of(text).is_some());
        assert!(looks_like_unparsed_tool_call(text));
    }

    #[test]
    fn malformed_bare_json_is_flagged_as_an_unparsed_attempt() {
        let text = r#"{"name": "board_attention", "arguments": {"#;
        assert!(tool_calls(text).is_empty());
        assert!(looks_like_unparsed_tool_call(text));
    }

    #[test]
    fn ordinary_prose_is_never_flagged_as_an_unparsed_attempt() {
        assert!(!looks_like_unparsed_tool_call("8 PRs await review."));
        assert!(!looks_like_unparsed_tool_call(
            "Sure - here's an example: {\"name\": \"x\"} in your config."
        ));
    }
}
