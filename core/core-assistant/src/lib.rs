//! Conversational assistant: a Rig agent over the local llama.cpp engine (`core-llm`), hybrid
//! retrieval (`core-store`/`core-embed`, exposed to Rig as a `VectorStoreIndex`), and read-only KB
//! tools. Off by default; the deterministic core does not depend on it.
//! The host calls [`answer`] with the board context and conversation. With the `assistant` feature
//! off (or no capable model), [`answer`] returns [`AssistantError::Disabled`].

use std::sync::Arc;

#[cfg(feature = "assistant")]
mod model;
#[cfg(feature = "assistant")]
mod prompt;
#[cfg(feature = "assistant")]
mod retrieval;
#[cfg(feature = "assistant")]
mod tools;

/// Why the assistant could not answer. The UI shows the reason, not a made-up answer.
#[derive(Debug)]
pub enum AssistantError {
    /// The `assistant` feature is off, or no capable model is configured. The default.
    Disabled,
    /// The engine or agent failed. Carries the message for logging and the Debug panel.
    Engine(String),
}

impl std::fmt::Display for AssistantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssistantError::Disabled => write!(f, "the AI assistant is disabled"),
            AssistantError::Engine(m) => write!(f, "AI assistant error: {m}"),
        }
    }
}

impl std::error::Error for AssistantError {}

/// One prior turn of the conversation. `role` is "user" or "assistant".
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Everything a single assistant turn needs: the model, the board scope, the conversation so far,
/// and the new question.
#[derive(Debug, Clone)]
pub struct AssistantRequest {
    pub model_dir: String,
    pub gguf_file: String,
    /// The board's effective repo ids; tools and retrieval are scoped to these.
    pub repo_ids: Vec<String>,
    /// Wall-clock now (RFC-3339), supplied by the host.
    pub now: String,
    /// A human label for the board, for the preamble.
    pub board_label: String,
    /// The tab the manager is currently looking at (e.g. "attention"), if on a board view.
    pub active_tab: Option<String>,
    pub history: Vec<ChatTurn>,
    pub question: String,
}

/// A streamed step of an assistant turn, handed to the host's callback.
#[derive(Debug, Clone)]
pub enum AssistantEvent {
    /// A piece of the answer text.
    Delta(String),
    /// The agent is calling a tool by this name.
    Tool(String),
    /// The turn finished.
    Done,
}

/// True when the `assistant` feature is compiled into this build.
pub fn feature_enabled() -> bool {
    cfg!(feature = "assistant")
}

/// Max tool-call + answer rounds before the agent stops.
#[cfg(feature = "assistant")]
const MAX_TURNS: usize = 6;

/// Build the agent's preamble: role, board context, behaviour rules, and the conversation so far.
#[cfg(feature = "assistant")]
fn preamble(req: &AssistantRequest) -> String {
    let mut p = format!(
        "You are orgonzola's engineering-intelligence assistant for the board \"{}\" ({} repositories), \
         helping an engineering manager understand THIS board.\n\
         \n\
         How to respond:\n\
         - If the message is a greeting, thanks, small talk, or unclear, reply briefly and naturally in one \
         sentence (greet back, or ask what they want to know about the board). Do NOT call any tool and do \
         NOT recite board state when no data was asked for.\n\
         - Only when the manager actually asks about the board - PRs, reviews, CI, issues, people, code, \
         activity - call the tools to fetch real data, and ground EVERY claim in tool results or the \
         provided context. Never invent numbers, names, PRs, issues, or facts. If the data does not answer \
         the question, say so plainly.\n\
         \n\
         Be concise and concrete; prefer specifics (PR numbers, counts, ages) over generalities.",
        req.board_label,
        req.repo_ids.len(),
    );
    if let Some(tab) = req.active_tab.as_deref().filter(|t| !t.is_empty()) {
        p.push_str(&format!(
            "\n\nContext: the manager is currently on the '{tab}' tab. If they ask about the board, lead \
             with what is relevant there when it fits - but this is context only, not a request to \
             summarize that tab."
        ));
    }
    if !req.history.is_empty() {
        p.push_str("\n\nConversation so far:");
        for turn in &req.history {
            p.push_str(&format!("\n{}: {}", turn.role, turn.content));
        }
    }
    p
}

/// Run one assistant turn over the local model, board-scoped tools and RAG, and return the answer.
#[cfg(feature = "assistant")]
pub async fn answer(
    store: Arc<core_store::Store>,
    embedder: Arc<dyn core_embed::Embedder>,
    req: AssistantRequest,
) -> Result<String, AssistantError> {
    use rig::completion::Prompt;

    let model = model::LocalModel::new(&req.model_dir, &req.gguf_file);
    let agent = tools::build_agent(
        model,
        store,
        embedder,
        req.repo_ids.clone(),
        req.now.clone(),
        preamble(&req),
    );
    // Max tool-call rounds are set on the agent (`default_max_turns`).
    agent
        .prompt(req.question.as_str())
        .await
        .map_err(|e| AssistantError::Engine(e.to_string()))
}

/// Stream one assistant turn, handing each answer-text delta, tool call and completion to
/// `on_event`. On engine failure, returns the error.
#[cfg(feature = "assistant")]
pub async fn answer_stream(
    store: Arc<core_store::Store>,
    embedder: Arc<dyn core_embed::Embedder>,
    req: AssistantRequest,
    mut on_event: impl FnMut(AssistantEvent),
) -> Result<(), AssistantError> {
    use futures::StreamExt;
    use rig::agent::MultiTurnStreamItem;
    use rig::completion::Message;
    use rig::streaming::{StreamedAssistantContent, StreamingPrompt};

    let model = model::LocalModel::new(&req.model_dir, &req.gguf_file);
    let agent = tools::build_agent(
        model,
        store,
        embedder,
        req.repo_ids.clone(),
        req.now.clone(),
        preamble(&req),
    );
    let history: Vec<Message> = req
        .history
        .iter()
        .map(|t| match t.role.as_str() {
            "assistant" => Message::assistant(t.content.clone()),
            "system" => Message::system(t.content.clone()),
            _ => Message::user(t.content.clone()),
        })
        .collect();

    let mut stream = agent
        .stream_prompt(req.question.as_str())
        .with_history(history.as_slice())
        .multi_turn(MAX_TURNS)
        .await;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(t))) => {
                on_event(AssistantEvent::Delta(t.text));
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            })) => {
                on_event(AssistantEvent::Tool(tool_call.function.name));
            }
            Ok(_) => {}
            Err(e) => return Err(AssistantError::Engine(e.to_string())),
        }
    }
    on_event(AssistantEvent::Done);
    Ok(())
}

/// The no-assistant build: always disabled. Keeps the host call site identical.
#[cfg(not(feature = "assistant"))]
pub async fn answer(
    _store: Arc<core_store::Store>,
    _embedder: Arc<dyn core_embed::Embedder>,
    _req: AssistantRequest,
) -> Result<String, AssistantError> {
    Err(AssistantError::Disabled)
}

#[cfg(not(feature = "assistant"))]
pub async fn answer_stream(
    _store: Arc<core_store::Store>,
    _embedder: Arc<dyn core_embed::Embedder>,
    _req: AssistantRequest,
    _on_event: impl FnMut(AssistantEvent),
) -> Result<(), AssistantError> {
    Err(AssistantError::Disabled)
}
