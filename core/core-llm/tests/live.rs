//! Live check (ignored by default): load the configured GGUF and complete a chat end to end. It
//! exercises the real model load and decode (and, with an accel feature, GPU offload and device
//! selection, which llama.cpp logs to stderr). Run it on a machine that has a model:
//!   ORGONZOLA_LLM_DIR="$PWD/hosts/desktop/resources/models/llm" \
//!   ORGONZOLA_LLM_GGUF="Qwen3-0.6B-Q4_K_M.gguf" \
//!     cargo test -p core-llm --features vulkan -- --ignored --nocapture
//! Drop `--features vulkan` (use `--features llm`) to check the CPU path.
#![cfg(feature = "llm")]

use core_llm::ChatMessage;

#[tokio::test]
#[ignore = "needs a fetched GGUF model + a real machine; run with --ignored"]
async fn live_complete() {
    let dir = std::env::var("ORGONZOLA_LLM_DIR").expect("set ORGONZOLA_LLM_DIR");
    let file = std::env::var("ORGONZOLA_LLM_GGUF").expect("set ORGONZOLA_LLM_GGUF");
    // The attention-narrator shape: a caller-owned system prompt + pressing items as user turn.
    let system = "You are an assistant to an engineering manager. From the attention items below, write two \
        or three short, plain sentences saying what to focus on first and why. Use ONLY the facts given.";
    let brief = "6 item(s) need attention: 1 failing CI, 3 awaiting review, 2 stale PR.\n\
        Most pressing items:\n\
        - [failing CI] CI red on PR #214 'cache invalidation' since this morning\n\
        - [awaiting review] PR #198 'auth refresh' open 6 days, no reviewer (your team)\n\
        - [awaiting review] PR #205 'docs tidy' open 2 days\n\
        - [stale PR] PR #140 'old refactor' open 19 days, no activity";
    let messages = vec![ChatMessage::system(system), ChatMessage::user(brief)];
    let out = core_llm::complete(&dir, &file, messages).await;
    println!("\n=== complete result ===\n{out:?}\n");
    assert!(out.is_ok(), "complete failed: {out:?}");
}

/// Prefill batch overflow ("Insufficient Space of 512"): a prompt longer than one decode batch
/// must be fed to `decode` in chunks. Builds a prompt well past 512 tokens and confirms the
/// completion runs.
#[tokio::test]
#[ignore = "needs a fetched GGUF model + a real machine; run with --ignored"]
async fn live_long_prompt_prefills_in_chunks() {
    let dir = std::env::var("ORGONZOLA_LLM_DIR").expect("set ORGONZOLA_LLM_DIR");
    let file = std::env::var("ORGONZOLA_LLM_GGUF").expect("set ORGONZOLA_LLM_GGUF");
    // ~2000 words of filler -> several thousand tokens, spanning many 512-token prefill batches
    // (still under the 8192 context).
    let filler =
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn. ".repeat(150);
    let messages = vec![
        ChatMessage::system("Reply with exactly the word: OK."),
        ChatMessage::user(format!("Context you can ignore:\n{filler}\nNow reply.")),
    ];
    let out = core_llm::complete(&dir, &file, messages).await;
    println!("\n=== long-prompt result ===\n{out:?}\n");
    assert!(out.is_ok(), "long-prompt complete failed: {out:?}");
}
