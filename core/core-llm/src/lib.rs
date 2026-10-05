//! Optional local-LLM completion. With the `llm` feature, [`complete`]/[`stream_complete`] run a
//! quantized GGUF via llama.cpp (`llama-cpp-2`) over caller-supplied [`ChatMessage`]s; the
//! caller owns its system prompt and history. Without the feature, or on any engine error, they
//! return [`LlmError`] and the caller keeps its deterministic output. A reasoning model's
//! `<think>` block is filtered out of the streamed text.
//! Engine: llama.cpp, because it ships a Vulkan backend (the only path to an Intel Arc GPU). The
//! accel features (`vulkan`/`cuda`/`metal`) offload to the GPU; plain `llm` is portable CPU.
//! Concurrency: `LlamaContext` is `!Send`. The model and backend (`Send + Sync`) live in a
//! process static; each request creates and consumes its context inside one `spawn_blocking`
//! closure and streams tokens back over a channel.

/// Why no completion was produced. The caller falls back to its deterministic output.
#[derive(Debug)]
pub enum LlmError {
    /// The `llm` feature is off, or no model is configured.
    Disabled,
    /// The engine failed (model load, generation). Carries the message for logging.
    Engine(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Disabled => write!(f, "local LLM narration is disabled"),
            LlmError::Engine(m) => write!(f, "local LLM engine error: {m}"),
        }
    }
}

impl std::error::Error for LlmError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One message in a chat completion.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

#[cfg(feature = "llm")]
mod engine {
    //! The llama.cpp engine, behind the `llm` feature.
    use super::{ChatMessage, LlmError};
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use std::num::NonZeroU32;
    use std::sync::{Mutex, OnceLock};

    /// Context window for the prompt plus generation.
    const N_CTX: u32 = 8192;
    /// Tokens per prefill `decode` call (the context's logical batch). Longer prompts are fed in
    /// chunks of this size. Must match the context's `n_batch` below.
    const PREFILL_BATCH: usize = 512;
    /// Cap on generated tokens. A reasoning model spends part of it inside `<think>` (filtered
    /// by [`ThinkFilter`]).
    const MAX_NEW_TOKENS: i32 = 1024;

    /// The loaded engine: backend token and model, both `Send + Sync`, in a process static. The
    /// `LlamaContext` is `!Send` and is created per request.
    struct Engine {
        backend: LlamaBackend,
        model: LlamaModel,
    }

    static ENGINE: OnceLock<Engine> = OnceLock::new();
    /// Serializes the one-time load.
    static LOAD_LOCK: Mutex<()> = Mutex::new(());
    static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

    /// Record and wrap an engine error so the Debug panel can show why a request failed.
    pub fn engine_err(e: impl ToString) -> LlmError {
        let s = e.to_string();
        if let Ok(mut g) = LAST_ERROR.lock() {
            *g = Some(s.clone());
        }
        LlmError::Engine(s)
    }

    /// Clear the recorded error at the start of an attempt.
    pub fn clear_last_error() {
        if let Ok(mut g) = LAST_ERROR.lock() {
            *g = None;
        }
    }

    pub fn last_error() -> Option<String> {
        LAST_ERROR.lock().ok().and_then(|g| g.clone())
    }

    pub fn is_loaded() -> bool {
        ENGINE.get().is_some()
    }

    /// All layers offload to the GPU when the Vulkan backend is built in, otherwise CPU.
    /// llama.cpp falls back to CPU if no Vulkan device is present.
    const fn n_gpu_layers() -> u32 {
        if cfg!(feature = "vulkan") {
            // High enough to offload every layer of any model we ship; llama.cpp clamps it.
            1000
        } else {
            0
        }
    }

    /// The loaded engine, built once and cached for the process (the GGUF load is slow). The
    /// first configured model wins; changing it needs a restart. Runs on the calling (blocking)
    /// thread.
    fn engine(model_dir: &str, gguf_file: &str) -> Result<&'static Engine, LlmError> {
        if let Some(e) = ENGINE.get() {
            return Ok(e);
        }
        let _guard = LOAD_LOCK.lock().map_err(engine_err)?;
        // Re-check under the lock: another thread may have loaded it while we waited.
        if let Some(e) = ENGINE.get() {
            return Ok(e);
        }
        // Stop llama.cpp/ggml from writing load-metadata and device-info lines to stderr: route
        // them through `tracing` with logs disabled. Must run before backend init to catch
        // device-probe output, and only once (the crate cannot re-init the log state).
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );
        let backend = LlamaBackend::init().map_err(engine_err)?;
        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers());
        let path = std::path::Path::new(model_dir).join(gguf_file);
        let model = LlamaModel::load_from_file(&backend, &path, &params).map_err(engine_err)?;
        // `set()` fails only if another thread won the race; ENGINE is populated either way.
        let _ = ENGINE.set(Engine { backend, model });
        Ok(ENGINE.get().expect("engine set above"))
    }

    /// Strips a leading `<think>...</think>` reasoning block from a token stream. Buffers the
    /// lead until it can tell whether a think block is present, suppresses everything through
    /// `</think>`, then passes the rest through. Tag-aware across piece boundaries.
    struct ThinkFilter {
        /// True once we know whether a think block is present.
        decided: bool,
        /// True while inside an open think block.
        open: bool,
        /// True once non-whitespace answer text has been emitted. Until then leading whitespace
        /// is dropped.
        started: bool,
        /// The lead held back until `decided`, or the think block until its close tag arrives.
        buf: String,
    }

    impl ThinkFilter {
        const OPEN: &'static str = "<think>";
        const CLOSE: &'static str = "</think>";

        fn new() -> Self {
            Self {
                decided: false,
                open: false,
                started: false,
                buf: String::new(),
            }
        }

        /// All emitted text goes through here: drops leading whitespace until the first real
        /// content.
        fn emit(&mut self, s: String) -> String {
            if self.started {
                return s;
            }
            let trimmed = s.trim_start();
            if trimmed.is_empty() {
                return String::new();
            }
            self.started = true;
            trimmed.to_string()
        }

        /// Feed one decoded piece; returns the text to emit now (often empty while buffering).
        fn push(&mut self, piece: &str) -> String {
            // Past the think block (or never had one): pass through, modulo leading whitespace.
            if self.decided && !self.open {
                return self.emit(piece.to_string());
            }
            self.buf.push_str(piece);
            if !self.decided {
                let lead = self.buf.trim_start();
                if lead.starts_with(Self::OPEN) {
                    self.decided = true;
                    self.open = true;
                } else if Self::OPEN.starts_with(lead) {
                    // Might still grow into "<think>": keep buffering.
                    return String::new();
                } else {
                    // No think block: flush the held lead.
                    self.decided = true;
                    let out = std::mem::take(&mut self.buf);
                    return self.emit(out);
                }
            }
            // Inside a think block: emit the remainder once the close tag is seen.
            if let Some(idx) = self.buf.find(Self::CLOSE) {
                self.open = false;
                let after = self.buf[idx + Self::CLOSE.len()..].to_string();
                self.buf.clear();
                return self.emit(after);
            }
            String::new()
        }

        /// End of stream: flush what is buffered. An unterminated think block yields nothing, so
        /// the caller falls back rather than show raw chain-of-thought.
        fn finish(&mut self) -> String {
            if self.open {
                String::new()
            } else {
                let out = std::mem::take(&mut self.buf);
                self.emit(out)
            }
        }
    }

    /// Run one generation against the loaded engine, calling `on_piece` with each decoded
    /// fragment (think block filtered out). Blocking; run it on a blocking thread. The prompt
    /// uses the model's own chat template. `on_piece` returns `false` to stop early.
    fn generate(
        e: &Engine,
        messages: &[ChatMessage],
        mut on_piece: impl FnMut(&str) -> bool,
    ) -> Result<(), LlmError> {
        let model = &e.model;
        let template = model.chat_template(None).map_err(engine_err)?;
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| {
                LlamaChatMessage::new(m.role.as_str().to_string(), m.content.clone())
                    .map_err(engine_err)
            })
            .collect::<Result<_, _>>()?;
        let prompt = model
            .apply_chat_template(&template, &chat, true)
            .map_err(engine_err)?;
        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(engine_err)?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_batch(PREFILL_BATCH as u32);
        let mut ctx = model
            .new_context(&e.backend, ctx_params)
            .map_err(engine_err)?;

        // Prefill in PREFILL_BATCH-sized chunks, requesting logits only on the last token. One
        // oversized batch overflows the context's n_batch ("Insufficient Space of 512").
        let n_prompt = tokens.len();
        let last = n_prompt - 1;
        let mut batch = LlamaBatch::new(PREFILL_BATCH, 1);
        for (chunk_start, chunk) in (0..n_prompt)
            .step_by(PREFILL_BATCH)
            .zip(tokens.chunks(PREFILL_BATCH))
        {
            batch.clear();
            for (j, &token) in chunk.iter().enumerate() {
                let pos = chunk_start + j;
                batch
                    .add(token, pos as i32, &[0], pos == last)
                    .map_err(engine_err)?;
            }
            ctx.decode(&mut batch).map_err(engine_err)?;
        }

        // Greedy decode, streaming each piece until EOG or the token cap. The last prefill token
        // carries the logits to sample from.
        let mut n_cur = n_prompt as i32;
        let stop = n_cur + MAX_NEW_TOKENS;
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut filter = ThinkFilter::new();
        while n_cur <= stop {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if model.is_eog_token(token) {
                break;
            }
            // special = false: plain text, not special-token markup.
            let piece = model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(engine_err)?;
            if !piece.is_empty() {
                let emit = filter.push(&piece);
                if !emit.is_empty() && !on_piece(&emit) {
                    return Ok(()); // consumer dropped: stop early
                }
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true).map_err(engine_err)?;
            n_cur += 1;
            ctx.decode(&mut batch).map_err(engine_err)?;
        }
        // Flush any buffered tail.
        let tail = filter.finish();
        if !tail.is_empty() {
            on_piece(&tail);
        }
        Ok(())
    }

    /// Load and cache the model without generating, so a caller can show a "loading" phase
    /// first. Runs the blocking load off the async runtime.
    pub async fn ensure_loaded(model_dir: &str, gguf_file: &str) -> Result<(), LlmError> {
        clear_last_error();
        let (dir, file) = (model_dir.to_string(), gguf_file.to_string());
        let res = tokio::task::spawn_blocking(move || engine(&dir, &file).map(|_| ())).await;
        match res {
            Ok(inner) => inner,
            Err(join) => Err(engine_err(join)),
        }
    }

    pub async fn complete(
        model_dir: &str,
        gguf_file: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LlmError> {
        clear_last_error();
        let (dir, file) = (model_dir.to_string(), gguf_file.to_string());
        let out = tokio::task::spawn_blocking(move || -> Result<String, LlmError> {
            let e = engine(&dir, &file)?;
            let mut acc = String::new();
            generate(e, &messages, |p| {
                acc.push_str(p);
                true
            })?;
            Ok(acc)
        })
        .await
        .map_err(engine_err)??;
        let out = out.trim().to_string();
        if out.is_empty() {
            return Err(engine_err("empty response"));
        }
        Ok(out)
    }

    /// Stream a completion: `on_token` is called with each text delta and returns `false` to
    /// stop early. Decoding runs on a blocking thread and sends pieces over an unbounded
    /// channel; `on_token` runs on the async side, so it need not be `Send`.
    pub async fn stream_complete(
        model_dir: &str,
        gguf_file: &str,
        messages: Vec<ChatMessage>,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<(), LlmError> {
        clear_last_error();
        let (dir, file) = (model_dir.to_string(), gguf_file.to_string());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let handle = tokio::task::spawn_blocking(move || -> Result<(), LlmError> {
            let e = engine(&dir, &file)?;
            // `send` fails once `rx` is dropped, which stops the `generate` decode loop.
            generate(e, &messages, |p| tx.send(p.to_string()).is_ok())
        });
        while let Some(piece) = rx.recv().await {
            if !on_token(&piece) {
                break;
            }
        }
        // Drop the receiver so the producer's next `send` fails, then await its result.
        drop(rx);
        match handle.await {
            Ok(inner) => inner,
            Err(join) => Err(engine_err(join)),
        }
    }

    #[cfg(test)]
    mod think_filter_tests {
        use super::ThinkFilter;

        fn run(pieces: &[&str]) -> String {
            let mut f = ThinkFilter::new();
            let mut out = String::new();
            for p in pieces {
                out.push_str(&f.push(p));
            }
            out.push_str(&f.finish());
            out
        }

        #[test]
        fn strips_a_think_block() {
            assert_eq!(
                run(&["<think>", "reasoning here", "</think>", "The answer."]),
                "The answer."
            );
        }

        #[test]
        fn strips_a_think_block_split_across_tokens() {
            // Tags arrive a few characters at a time.
            let out = run(&[
                "<", "think", ">", "ponder", "</", "think", ">", "\n\n", "Done.",
            ]);
            assert_eq!(out, "Done.");
        }

        #[test]
        fn passes_through_when_there_is_no_think_block() {
            assert_eq!(run(&["Three", " PRs", " await."]), "Three PRs await.");
        }

        #[test]
        fn unterminated_think_block_yields_nothing() {
            // No close tag: the answer never came, so emit nothing.
            assert_eq!(run(&["<think>", "still thinking and then cut off"]), "");
        }

        #[test]
        fn handles_content_immediately_after_close_in_one_piece() {
            assert_eq!(run(&["<think>x</think>The answer."]), "The answer.");
        }
    }
}

/// True when the `llm` feature is compiled into this build.
pub fn feature_enabled() -> bool {
    cfg!(feature = "llm")
}

/// Whether the model has been loaded this session.
pub fn is_loaded() -> bool {
    #[cfg(feature = "llm")]
    {
        engine::is_loaded()
    }
    #[cfg(not(feature = "llm"))]
    {
        false
    }
}

/// The most recent engine error, for the Debug panel; `None` if the last attempt succeeded.
pub fn last_error() -> Option<String> {
    #[cfg(feature = "llm")]
    {
        engine::last_error()
    }
    #[cfg(not(feature = "llm"))]
    {
        None
    }
}

/// One-shot chat completion over `messages` using a local GGUF model from
/// `model_dir`/`gguf_file`. On failure the caller should fall back to its deterministic output.
#[cfg(feature = "llm")]
pub async fn complete(
    model_dir: &str,
    gguf_file: &str,
    messages: Vec<ChatMessage>,
) -> Result<String, LlmError> {
    engine::complete(model_dir, gguf_file, messages).await
}

/// Load and cache the model without generating. Cheap once loaded.
#[cfg(feature = "llm")]
pub async fn ensure_loaded(model_dir: &str, gguf_file: &str) -> Result<(), LlmError> {
    engine::ensure_loaded(model_dir, gguf_file).await
}

/// Stream a chat completion: `on_token` is called with each text delta. Loads and caches the
/// model first. Errors (including the feature being off) mean the caller falls back.
#[cfg(feature = "llm")]
pub async fn stream_complete(
    model_dir: &str,
    gguf_file: &str,
    messages: Vec<ChatMessage>,
    on_token: impl FnMut(&str) -> bool,
) -> Result<(), LlmError> {
    engine::stream_complete(model_dir, gguf_file, messages, on_token).await
}

/// The no-LLM build: always falls back.
#[cfg(not(feature = "llm"))]
pub async fn complete(
    _model_dir: &str,
    _gguf_file: &str,
    _messages: Vec<ChatMessage>,
) -> Result<String, LlmError> {
    Err(LlmError::Disabled)
}

#[cfg(not(feature = "llm"))]
pub async fn ensure_loaded(_model_dir: &str, _gguf_file: &str) -> Result<(), LlmError> {
    Err(LlmError::Disabled)
}

/// The no-LLM build of the streaming completion: reports `Disabled`.
#[cfg(not(feature = "llm"))]
pub async fn stream_complete(
    _model_dir: &str,
    _gguf_file: &str,
    _messages: Vec<ChatMessage>,
    _on_token: impl FnMut(&str) -> bool,
) -> Result<(), LlmError> {
    Err(LlmError::Disabled)
}

// Only the no-`llm` build has a "disabled" path. Run with `cargo test -p core-llm
// --no-default-features`.
#[cfg(all(test, not(feature = "llm")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn complete_is_disabled_without_the_feature() {
        // Without the `llm` feature, `complete` always falls back.
        let r = complete("dir", "model.gguf", vec![ChatMessage::user("hi")]).await;
        assert!(matches!(r, Err(LlmError::Disabled)));
    }
}
