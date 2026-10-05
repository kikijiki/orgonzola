//! A Rig [`CompletionModel`] backed by the in-process `core-llm` engine (llama.cpp). Rig's agent
//! drives tool calling and RAG; every model call routes to the on-device GGUF.
//! `stream()` streams answer text as produced, but buffers a `<tool_call>` turn and parses it at
//! the end, since tokens that might become a tool call cannot be retracted.

use crate::prompt::{
    classify_json_object, looks_like_unparsed_tool_call, parse_choice, render_messages, JsonShape,
};
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
use rig::message::AssistantContent;
use rig::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse};

/// The model directory and filename `core-llm` loads (and caches) on first use. Cheap to clone.
#[derive(Clone)]
pub struct LocalModel {
    pub model_dir: String,
    pub gguf_file: String,
}

impl LocalModel {
    pub fn new(model_dir: impl Into<String>, gguf_file: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            gguf_file: gguf_file.into(),
        }
    }
}

/// Decides, as tokens arrive, whether this turn is an answer (stream it) or a `<tool_call>`
/// (buffer it). `core-llm` has already stripped any `<think>` block.
#[derive(Default)]
struct StreamGate {
    decided: bool,
    tool: bool,
    buf: String,
}

/// What is left in the gate at end of stream.
enum GateEnd {
    /// Answer text (the held-back tail / a short answer) to emit.
    Text(String),
    /// The buffered `<tool_call>` block(s) to parse into tool calls.
    Tool(String),
}

fn non_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// If `s` starts with `{` and holds a complete, brace-balanced top-level JSON object, return its
/// byte length. `None` means not closed yet (keep buffering). Tracks string/escape state so a `}`
/// inside a quoted value does not end the object; only indexes at ASCII bytes, so the offset is a
/// valid `str` boundary.
fn json_object_len(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

impl StreamGate {
    const OPEN: &'static str = "<tool_call>";

    /// Feed a piece; returns answer text to stream now. `None` while undecided, buffering a tool
    /// call, or holding back a short tail that might grow into a tool call.
    fn push(&mut self, piece: &str) -> Option<String> {
        self.buf.push_str(piece);
        if !self.decided {
            let lead = self.buf.trim_start();
            if lead.starts_with(Self::OPEN) {
                self.decided = true;
                self.tool = true;
                return None;
            }
            if Self::OPEN.starts_with(lead) {
                // Could still grow into "<tool_call>"; keep buffering.
                return None;
            }
            if lead.starts_with('{') {
                // Might be a bare (unwrapped) tool call: buffer until the object closes. A closed
                // object that is not call-shaped is JSON prose and streams; a malformed one is
                // left for `finish_tool_turn` to report.
                let len = json_object_len(lead)?; // not closed yet; keep buffering
                self.decided = true;
                self.tool = !matches!(classify_json_object(&lead[..len]), JsonShape::Other);
                return None;
            }
            self.decided = true; // it does not start with the tool tag or a brace, so it is an answer
        }
        if self.tool {
            return None; // buffering the rest of a tool-call turn for end-of-stream parsing
        }
        // Answer mode. A reasoning model can emit a tool call after prose, so keep scanning for
        // the tag and switch to tool mode when it appears.
        if let Some(idx) = self.buf.find(Self::OPEN) {
            self.tool = true;
            let rest = self.buf.split_off(idx); // self.buf = prose before the tag
            let prose = std::mem::replace(&mut self.buf, rest); // keep the tag onward for parsing
            return non_empty(prose);
        }
        // No tag yet: stream all but a held-back tail so a tag split across pieces is caught.
        // Split on a char boundary so multi-byte UTF-8 does not panic.
        let keep = Self::OPEN.len() - 1;
        if self.buf.len() <= keep {
            return None;
        }
        let mut split = self.buf.len() - keep;
        while split > 0 && !self.buf.is_char_boundary(split) {
            split -= 1;
        }
        let rest = self.buf.split_off(split);
        let out = std::mem::replace(&mut self.buf, rest);
        non_empty(out)
    }

    /// Drain what is left at end of stream.
    fn finish(&mut self) -> GateEnd {
        let buf = std::mem::take(&mut self.buf);
        if self.tool {
            GateEnd::Tool(buf)
        } else {
            GateEnd::Text(buf)
        }
    }
}

impl CompletionModel for LocalModel {
    type Response = ();
    type StreamingResponse = ();
    type Client = ();

    /// Rig constructs models from an HTTP client; ours is built via [`LocalModel::new`], so `make`
    /// is never reached. Present to satisfy the trait.
    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        unreachable!(
            "LocalModel is constructed directly via LocalModel::new, not from a provider client"
        )
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let messages = render_messages(&request);
        let text = core_llm::complete(&self.model_dir, &self.gguf_file, messages)
            .await
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?;
        let choice = parse_choice(&text);
        // Plain text that still looks like a botched tool call is an error, not an answer.
        if choice
            .iter()
            .all(|c| matches!(c, AssistantContent::Text(_)))
            && looks_like_unparsed_tool_call(&text)
        {
            return Err(CompletionError::ResponseError(format!(
                "the model attempted a tool call it did not format correctly: {}",
                text.trim()
            )));
        }
        Ok(CompletionResponse {
            choice,
            usage: Usage::new(),
            raw_response: (),
            message_id: None,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let messages = render_messages(&request);
        let (dir, file) = (self.model_dir.clone(), self.gguf_file.clone());
        let (tx, rx) =
            futures::channel::mpsc::unbounded::<Result<RawStreamingChoice<()>, CompletionError>>();

        tokio::spawn(async move {
            let mut gate = StreamGate::default();
            // Returns `true` to keep generating. Once the Rig stream is dropped, `tx` closes and
            // returning `false` stops the decode loop.
            let emit = |piece: &str| -> bool {
                if let Some(text) = gate.push(piece) {
                    let _ = tx.unbounded_send(Ok(RawStreamingChoice::Message(text)));
                }
                !tx.is_closed()
            };
            let result = core_llm::stream_complete(&dir, &file, messages, emit).await;

            match result {
                Ok(()) => {
                    match gate.finish() {
                        // Tool-call turn: parse into Rig tool calls, or report a failed attempt.
                        GateEnd::Tool(buf) => match finish_tool_turn(&buf) {
                            Ok(calls) => {
                                for tc in calls {
                                    let _ = tx.unbounded_send(Ok(RawStreamingChoice::ToolCall(
                                        RawStreamingToolCall::new(
                                            tc.id,
                                            tc.function.name,
                                            tc.function.arguments,
                                        ),
                                    )));
                                }
                            }
                            Err(msg) => {
                                let _ = tx.unbounded_send(Err(CompletionError::ResponseError(msg)));
                            }
                        },
                        // Emit the tail, unless it looks like an unparsed tool call (then error).
                        GateEnd::Text(buf) => match finish_text_turn(buf) {
                            Ok(Some(text)) => {
                                let _ = tx.unbounded_send(Ok(RawStreamingChoice::Message(text)));
                            }
                            Ok(None) => {}
                            Err(msg) => {
                                let _ = tx.unbounded_send(Err(CompletionError::ResponseError(msg)));
                            }
                        },
                    }
                    let _ = tx.unbounded_send(Ok(RawStreamingChoice::FinalResponse(())));
                }
                Err(e) => {
                    let _ = tx.unbounded_send(Err(CompletionError::RequestError(Box::new(e))));
                }
            }
        });

        Ok(StreamingCompletionResponse::stream(Box::pin(rx)))
    }
}

/// Error message for a buffered attempt that never became a usable tool call or answer.
fn unparsed_attempt_message(raw: &str) -> String {
    format!(
        "the model attempted a tool call it did not format correctly: {}",
        raw.trim()
    )
}

/// Decide what to do with a buffered tool-call turn (`GateEnd::Tool`): the parsed tool calls, or
/// an error if the text never parsed into one. Pure, so it is unit-testable without a model.
fn finish_tool_turn(buf: &str) -> Result<Vec<rig::completion::message::ToolCall>, String> {
    let calls: Vec<_> = parse_choice(buf)
        .into_iter()
        .filter_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect();
    if calls.is_empty() {
        Err(unparsed_attempt_message(buf))
    } else {
        Ok(calls)
    }
}

/// Decide what to do with a buffered answer turn (`GateEnd::Text`): the text to stream (`None` if
/// whitespace only), or an error if it still looks like an unparsed tool call.
fn finish_text_turn(buf: String) -> Result<Option<String>, String> {
    if buf.trim().is_empty() {
        return Ok(None);
    }
    if looks_like_unparsed_tool_call(&buf) {
        Err(unparsed_attempt_message(&buf))
    } else {
        Ok(Some(buf))
    }
}

#[cfg(test)]
mod gate_tests {
    use super::{finish_text_turn, finish_tool_turn, GateEnd, StreamGate};

    /// Feed pieces through the gate; return (streamed text, buffered tool-call block if any).
    fn run(pieces: &[&str]) -> (String, Option<String>) {
        let mut g = StreamGate::default();
        let mut text = String::new();
        for p in pieces {
            if let Some(t) = g.push(p) {
                text.push_str(&t);
            }
        }
        match g.finish() {
            GateEnd::Text(b) => {
                text.push_str(&b);
                (text, None)
            }
            GateEnd::Tool(b) => (text, Some(b)),
        }
    }

    #[test]
    fn pure_answer_streams_in_full() {
        let (text, tool) = run(&["Three ", "PRs ", "await ", "review."]);
        assert_eq!(text, "Three PRs await review.");
        assert!(tool.is_none());
    }

    #[test]
    fn pure_tool_call_is_buffered_not_streamed() {
        let (text, tool) = run(&["<tool_call>", "{\"name\": \"x\"}", "</tool_call>"]);
        assert_eq!(text, "");
        assert!(tool.unwrap().contains("\"name\": \"x\""));
    }

    #[test]
    fn prose_then_tool_call_streams_prose_and_keeps_the_call() {
        // A reasoning model can explain, then call a tool, in one turn. The prose streams and the
        // tool call is captured.
        let (text, tool) = run(&["Let me check ", "<tool_call>{\"name\": \"a\"}</tool_call>"]);
        assert_eq!(text, "Let me check ");
        assert!(tool.unwrap().contains("\"name\": \"a\""));
    }

    #[test]
    fn tool_tag_split_across_pieces_is_caught() {
        let (text, tool) = run(&["<", "tool", "_call>", "{\"name\": \"b\"}", "</tool_call>"]);
        assert_eq!(text, "");
        assert!(tool.unwrap().contains("\"name\": \"b\""));
    }

    #[test]
    fn bare_json_tool_call_is_buffered_not_streamed_character_by_character() {
        // No wrapper tags at all.
        let (text, tool) = run(&["{\"name\": \"board_attention\", ", "\"arguments\": {}}"]);
        assert_eq!(text, "");
        assert!(tool.unwrap().contains("board_attention"));
    }

    #[test]
    fn bare_json_split_mid_string_value_is_not_ended_early_by_a_brace_in_the_value() {
        // The argument contains a literal "}"; a quoted brace must not close the object, even when
        // the string spans two pieces.
        let (text, tool) = run(&[
            "{\"name\": \"search_code\", \"arguments\": {\"query\": \"a } b\"",
            "}}",
        ]);
        assert_eq!(text, "");
        assert!(tool.unwrap().contains("search_code"));
    }

    #[test]
    fn json_object_without_a_name_field_streams_as_ordinary_text() {
        // Structured prose (no name/tool_call key) is not treated as a call attempt.
        let (text, tool) = run(&["{\"open_prs\": 3}", " is what I found."]);
        assert_eq!(text, "{\"open_prs\": 3} is what I found.");
        assert!(tool.is_none());
    }

    #[test]
    fn malformed_bare_json_is_buffered_not_streamed_as_prose() {
        let (text, tool) = run(&["{\"name\": \"x\", ", "\"arguments\": {bad}}"]);
        assert_eq!(text, "");
        assert!(tool.is_some()); // buffered for finish_tool_turn to report as a failed attempt, not shown
    }

    #[test]
    fn finish_tool_turn_extracts_the_call() {
        let calls = finish_tool_turn("{\"name\": \"board_attention\", \"arguments\": {}}").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "board_attention");
    }

    #[test]
    fn finish_tool_turn_errors_instead_of_faking_or_dropping_a_malformed_attempt() {
        let err = finish_tool_turn("{\"name\": \"x\", \"arguments\": {bad}}").unwrap_err();
        assert!(err.contains("did not format correctly"));
    }

    #[test]
    fn finish_text_turn_passes_through_ordinary_prose() {
        assert_eq!(
            finish_text_turn("Three PRs await review.".to_string()).unwrap(),
            Some("Three PRs await review.".to_string())
        );
    }

    #[test]
    fn finish_text_turn_drops_whitespace_only_output() {
        assert_eq!(finish_text_turn("   ".to_string()).unwrap(), None);
    }

    #[test]
    fn finish_text_turn_errors_when_text_still_looks_like_a_tool_call() {
        // Not reachable with the current gate (a stray tag decides "tool" up front); this is a
        // safety net re-checking whatever reaches GateEnd::Text.
        assert!(finish_text_turn("<tool_call>{\"name\": \"x\"}".to_string()).is_err());
    }
}
