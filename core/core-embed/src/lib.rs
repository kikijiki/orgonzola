//! Embedding for semantic search: the `Embedder` trait, a deterministic `HashEmbedder`, and
//! AST-aware code chunking (tree-sitter via `text-splitter`). A real model (fastembed) is a
//! feature-gated impl, so the pipeline builds and tests with no ML model, network or display.

use serde::{Deserialize, Serialize};

/// Maps text to a fixed-length vector. Deterministic: the same text always yields the same
/// vector. `Send + Sync` so an `Arc<dyn Embedder>` can be shared across async tasks.
pub trait Embedder: Send + Sync {
    fn dimensions(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Short identifier of the active embedder ("hash" or the model name), for status readouts.
    fn name(&self) -> &str;
}

/// Bag-of-words embedder: tokenize, hash each token into a bucket, count, L2-normalize. Related
/// texts that share words land near each other; synonyms are not captured.
pub struct HashEmbedder {
    dimensions: usize,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0, "embedding dimensions must be > 0");
        Self { dimensions }
    }
}

/// The embedding dimension of the whole pipeline. The store's `vec0` table is fixed at this
/// width, so every embedder must produce vectors of this length.
pub const EMBED_DIM: usize = 768;

/// Version of the chunking scheme. Bump it when chunking changes enough to make an existing
/// index stale; the sync engine folds it into the recorded embedder identity, so the next sync
/// rebuilds.
pub const INDEX_SCHEME_VERSION: u32 = 2;

impl Default for HashEmbedder {
    /// `EMBED_DIM` dimensions, matching the vector store.
    fn default() -> Self {
        Self::new(EMBED_DIM)
    }
}

impl Embedder for HashEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dimensions];
        for token in tokenize(text) {
            let bucket = (fnv1a(&token) as usize) % self.dimensions;
            v[bucket] += 1.0;
        }
        l2_normalize(&mut v);
        v
    }

    fn name(&self) -> &str {
        "hash (deterministic)"
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// FNV-1a (64-bit). Fixed algorithm, so bucketing is stable across runs and machines, unlike
/// `std`'s `DefaultHasher`.
fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Stable content hash of `text` (FNV-1a, 64-bit), so the index can skip re-embedding unchanged
/// text.
pub fn content_hash(text: &str) -> u64 {
    fnv1a(text)
}

/// Scale a vector to unit length in place. A zero vector is left as zeros.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two equal-length vectors, in [-1, 1]. 0 if either is zero or lengths
/// differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    pub text: String,
}

/// Split `text` into chunks of at most `max_chars` characters on whitespace. A word longer than
/// `max_chars` becomes its own oversized chunk. Blank input yields no chunks.
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<Chunk> {
    let max = max_chars.max(1);
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= max {
            current.push(' ');
            current.push_str(word);
        } else {
            chunks.push(Chunk {
                text: std::mem::take(&mut current),
            });
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        chunks.push(Chunk { text: current });
    }
    chunks
}

/// Classify a repository path as source code, returning a coarse language label, or `None` to
/// skip it (lockfiles, data, images, binaries). Case-insensitive on the final extension.
pub fn source_language(path: &str) -> Option<&'static str> {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    // Non-source filenames that carry a source-looking extension or none.
    const EXCLUDED_NAMES: [&str; 4] = ["package-lock.json", "cargo.lock", "go.sum", "yarn.lock"];
    if EXCLUDED_NAMES.contains(&name.as_str()) {
        return None;
    }
    let ext = name.rsplit_once('.').map(|(_, e)| e)?;
    let lang = match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "rb" => "ruby",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hh" => "cpp",
        "cs" => "csharp",
        "swift" => "swift",
        "php" => "php",
        "scala" => "scala",
        "sh" | "bash" => "shell",
        "sql" => "sql",
        _ => return None,
    };
    Some(lang)
}

/// Target chunk size in characters for code and prose (~4 chars/token).
pub const CHUNK_CHARS: usize = 1500;

/// Target floor for a code chunk. text-splitter reads a capacity as `desired..max` and merges
/// sibling AST nodes until `desired`; without a floor it emits one-token fragments (`{`, `const`).
pub const CHUNK_MIN: usize = 500;

/// Lines per window for the line-based fallback when no grammar is available.
const CODE_FALLBACK_LINES: usize = 60;

/// Chunk source code along syntax-tree boundaries using tree-sitter. `lang` is a
/// `source_language` label; without a grammar (or on parse failure) it falls back to fixed line
/// windows. Blank chunks are dropped.
pub fn chunk_code(text: &str, lang: &str) -> Vec<Chunk> {
    if let Some(language) = tree_sitter_language(lang) {
        if let Ok(splitter) =
            text_splitter::CodeSplitter::new(language.clone(), CHUNK_MIN..CHUNK_CHARS)
        {
            // Parse once so each chunk is prefixed with its enclosing scope chain (impl/class/fn
            // names), so a query like "widget render" matches a `render` method inside `impl
            // Widget`.
            let tree = {
                let mut parser = tree_sitter::Parser::new();
                parser
                    .set_language(&language)
                    .ok()
                    .and_then(|()| parser.parse(text, None))
            };
            let comment = line_comment(lang);
            let mut out = Vec::new();
            for (offset, chunk) in splitter.chunk_indices(text) {
                if is_degenerate_code_chunk(chunk) {
                    continue;
                }
                let text = match tree
                    .as_ref()
                    .and_then(|t| scope_breadcrumb(t, text, offset))
                {
                    Some(crumb) => format!("{comment} {crumb}\n{chunk}"),
                    None => chunk.to_string(),
                };
                out.push(Chunk { text });
            }
            return out;
        }
    }
    chunk_code_lines(text, CODE_FALLBACK_LINES)
}

/// Whether a chunk is too small to be a useful retrieval unit: fewer than two word-like tokens
/// (>= 2 chars) or very short, e.g. `{`, `const`, `mod tests`. `fn parse_manifest() {}` is kept.
fn is_degenerate_code_chunk(chunk: &str) -> bool {
    let words = chunk
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .take(2)
        .count();
    words < 2 || chunk.trim().len() < 12
}

/// The enclosing scope chain at byte `offset`, outermost first, e.g. `"Widget > render"`. `None`
/// when there is no enclosing declaration. The name comes from the node's `name` or `type` field.
fn scope_breadcrumb(tree: &tree_sitter::Tree, text: &str, offset: usize) -> Option<String> {
    let mut node = tree.root_node().descendant_for_byte_range(offset, offset);
    let mut names = Vec::new();
    while let Some(n) = node {
        if is_decl_kind(n.kind()) {
            if let Some(name) = decl_name(&n, text) {
                names.push(name);
            }
        }
        node = n.parent();
    }
    if names.is_empty() {
        return None;
    }
    names.reverse(); // collected innermost-first
    Some(names.join(" > "))
}

/// Whether a tree-sitter node kind names a declaration for the breadcrumb. Substring match, so
/// it works across grammars.
fn is_decl_kind(kind: &str) -> bool {
    const MARKERS: [&str; 10] = [
        "function",
        "method",
        "class",
        "impl",
        "struct",
        "enum",
        "interface",
        "trait",
        "namespace",
        "mod",
    ];
    MARKERS.iter().any(|m| kind.contains(m))
}

/// The declared name of a declaration node: its `name` or `type` field (a Rust `impl Widget`).
fn decl_name(node: &tree_sitter::Node, text: &str) -> Option<String> {
    let name_node = node
        .child_by_field_name("name")
        .or_else(|| node.child_by_field_name("type"))?;
    let name = text.get(name_node.byte_range())?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// Line-comment token for a language, so the breadcrumb reads as a comment in the chunk.
fn line_comment(lang: &str) -> &'static str {
    match lang {
        "python" | "ruby" | "shell" => "#",
        _ => "//",
    }
}

/// Chunk prose (PR/issue bodies) along Markdown structure. Non-Markdown text falls back to plain
/// semantic splitting.
pub fn chunk_markdown(text: &str) -> Vec<Chunk> {
    let splitter = text_splitter::MarkdownSplitter::new(CHUNK_CHARS);
    collect_nonblank(splitter.chunks(text))
}

/// Map a `source_language` label to its tree-sitter grammar, or `None` (those chunk by line
/// window). Public because `core-codehealth` parses with the same grammars.
pub fn tree_sitter_language(lang: &str) -> Option<tree_sitter::Language> {
    let language = match lang {
        "rust" => tree_sitter_rust::LANGUAGE,
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        "javascript" => tree_sitter_javascript::LANGUAGE,
        "python" => tree_sitter_python::LANGUAGE,
        "go" => tree_sitter_go::LANGUAGE,
        "java" => tree_sitter_java::LANGUAGE,
        "c" => tree_sitter_c::LANGUAGE,
        "cpp" => tree_sitter_cpp::LANGUAGE,
        "ruby" => tree_sitter_ruby::LANGUAGE,
        "csharp" => tree_sitter_c_sharp::LANGUAGE,
        "php" => tree_sitter_php::LANGUAGE_PHP,
        "shell" => tree_sitter_bash::LANGUAGE,
        _ => return None,
    };
    Some(language.into())
}

fn collect_nonblank<'a>(chunks: impl Iterator<Item = &'a str>) -> Vec<Chunk> {
    chunks
        .filter(|c| !c.trim().is_empty())
        .map(|c| Chunk {
            text: c.to_string(),
        })
        .collect()
}

/// Line-window fallback: contiguous windows of at most `max_lines` whole lines. Blank windows
/// dropped.
fn chunk_code_lines(text: &str, max_lines: usize) -> Vec<Chunk> {
    let max = max_lines.max(1);
    let mut chunks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    for window in lines.chunks(max) {
        if window.iter().all(|l| l.trim().is_empty()) {
            continue;
        }
        chunks.push(Chunk {
            text: window.join("\n"),
        });
    }
    chunks
}

/// Reorders candidate documents by relevance to a query. A cross-encoder scores each (query,
/// document) pair jointly, so it is applied as a second stage over the fused top-K.
pub trait Reranker: Send + Sync {
    /// Return the indices of `documents` reordered best-first, a permutation of
    /// `0..documents.len()`.
    fn rerank(&self, query: &str, documents: &[String]) -> Vec<usize>;
    /// A short identifier of the active reranker ("none" for the no-op).
    fn name(&self) -> &str;
}

/// Keeps the input order. The default when no model is present.
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, _query: &str, documents: &[String]) -> Vec<usize> {
        (0..documents.len()).collect()
    }

    fn name(&self) -> &str {
        "none"
    }
}

/// Code embedding model (jina-embeddings-v2-base-code, `EMBED_DIM` dims) behind `Embedder`,
/// enabled by the `fastembed` feature. Off by default because it pulls in onnxruntime.
#[cfg(feature = "fastembed")]
pub struct FastEmbedder {
    // fastembed inference takes `&mut self` but `Embedder` is `&self` and shared via `Arc`, so
    // the model sits behind a Mutex.
    model: std::sync::Mutex<fastembed::TextEmbedding>,
}

/// Files `from_path` expects in a model directory: the ONNX graph plus four tokenizer JSON files.
#[cfg(feature = "fastembed")]
pub const MODEL_FILES: [&str; 5] = [
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

#[cfg(feature = "fastembed")]
#[derive(Debug, thiserror::Error)]
pub enum FastEmbedError {
    #[error("reading model file {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error(transparent)]
    Model(#[from] fastembed::Error),
}

#[cfg(feature = "fastembed")]
impl FastEmbedder {
    /// Load the model from a directory of bundled files (`MODEL_FILES`), with no network access.
    /// The production path.
    pub fn from_path(dir: &std::path::Path) -> Result<Self, FastEmbedError> {
        let tokenizer_files = read_tokenizer_files(dir)?;
        let onnx = read_model_file(dir, "model.onnx")?;
        let user_model = fastembed::UserDefinedEmbeddingModel::new(onnx, tokenizer_files);
        let model = fastembed::TextEmbedding::try_new_from_user_defined(
            user_model,
            fastembed::InitOptionsUserDefined::default(),
        )?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
        })
    }

    /// Load the model by downloading it on first run. Developer convenience; the shell uses
    /// `from_path`.
    pub fn new() -> Result<Self, FastEmbedError> {
        let model = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(fastembed::EmbeddingModel::JinaEmbeddingsV2BaseCode)
                .with_show_download_progress(false),
        )?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
        })
    }
}

#[cfg(feature = "fastembed")]
impl Embedder for FastEmbedder {
    fn dimensions(&self) -> usize {
        EMBED_DIM
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // One document in, one vector out. On failure (model error, poisoned lock) return a zero
        // vector rather than panic, so one bad input does not take down a sync.
        self.model
            .lock()
            .ok()
            .and_then(|mut model| model.embed(vec![text], None).ok())
            .and_then(|mut v| v.drain(..).next())
            .unwrap_or_else(|| vec![0.0; EMBED_DIM])
    }

    fn name(&self) -> &str {
        "jina-embeddings-v2-base-code"
    }
}

#[cfg(feature = "fastembed")]
fn read_model_file(dir: &std::path::Path, name: &str) -> Result<Vec<u8>, FastEmbedError> {
    let path = dir.join(name);
    std::fs::read(&path).map_err(|source| FastEmbedError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Read the four tokenizer JSON files of an ONNX model directory (shared by embedder and reranker).
#[cfg(feature = "fastembed")]
fn read_tokenizer_files(
    dir: &std::path::Path,
) -> Result<fastembed::TokenizerFiles, FastEmbedError> {
    Ok(fastembed::TokenizerFiles {
        tokenizer_file: read_model_file(dir, "tokenizer.json")?,
        config_file: read_model_file(dir, "config.json")?,
        special_tokens_map_file: read_model_file(dir, "special_tokens_map.json")?,
        tokenizer_config_file: read_model_file(dir, "tokenizer_config.json")?,
    })
}

/// Cross-encoder reranker (jina-reranker-v1-turbo-en) behind `Reranker`, enabled by the
/// `fastembed` feature. Loaded from bundled files like `FastEmbedder`.
#[cfg(feature = "fastembed")]
pub struct CrossEncoderReranker {
    // Behind a Mutex for the same reason as `FastEmbedder`.
    model: std::sync::Mutex<fastembed::TextRerank>,
}

#[cfg(feature = "fastembed")]
impl CrossEncoderReranker {
    /// Load the reranker from a directory of bundled model files (`MODEL_FILES`), no network
    /// access.
    pub fn from_path(dir: &std::path::Path) -> Result<Self, FastEmbedError> {
        let tokenizer_files = read_tokenizer_files(dir)?;
        let onnx = read_model_file(dir, "model.onnx")?;
        let user_model = fastembed::UserDefinedRerankingModel::new(onnx, tokenizer_files);
        let model = fastembed::TextRerank::try_new_from_user_defined(
            user_model,
            fastembed::RerankInitOptionsUserDefined::default(),
        )?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
        })
    }

    /// Load the model by downloading it on first run. Developer convenience; the shell uses
    /// `from_path`.
    pub fn new() -> Result<Self, FastEmbedError> {
        let model = fastembed::TextRerank::try_new(
            fastembed::RerankInitOptions::new(fastembed::RerankerModel::JINARerankerV1TurboEn)
                .with_show_download_progress(false),
        )?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
        })
    }
}

#[cfg(feature = "fastembed")]
impl Reranker for CrossEncoderReranker {
    fn rerank(&self, query: &str, documents: &[String]) -> Vec<usize> {
        // Score every (query, document) pair; results come back sorted best-first with their
        // original index. On failure fall back to the input order.
        let docs: Vec<&str> = documents.iter().map(String::as_str).collect();
        self.model
            .lock()
            .ok()
            .and_then(|mut model| model.rerank(query, docs, false, None).ok())
            .map(|results| results.into_iter().map(|r| r.index).collect())
            .unwrap_or_else(|| (0..documents.len()).collect())
    }

    fn name(&self) -> &str {
        "jina-reranker-v1-turbo-en"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_is_deterministic_and_fixed_length() {
        let e = HashEmbedder::new(64);
        let a = e.embed("the quick brown fox");
        let b = e.embed("the quick brown fox");
        assert_eq!(a.len(), 64);
        assert_eq!(a, b);
    }

    #[test]
    fn related_text_scores_above_unrelated() {
        // A query closer in words ranks higher.
        let e = HashEmbedder::default();
        let auth = e.embed("user authentication login flow");
        let migration = e.embed("database schema migration rollback");
        let query = e.embed("login authentication for a user");
        assert!(
            cosine_similarity(&query, &auth) > cosine_similarity(&query, &migration),
            "auth chunk should rank above the migration chunk"
        );
    }

    #[test]
    fn identical_text_is_maximally_similar() {
        let e = HashEmbedder::default();
        let v = e.embed("orgonzola dependency graph");
        // Normalized vector dotted with itself is ~1.0.
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn chunking_respects_the_size_bound_and_keeps_words_whole() {
        let text = "alpha beta gamma delta epsilon zeta";
        let chunks = chunk_text(text, 11); // "alpha beta" = 10, + " gamma" would exceed
        assert!(chunks.iter().all(|c| c.text.chars().count() <= 11));
        let rejoined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("   ", 100).is_empty());
    }

    #[test]
    fn hash_embedder_default_matches_the_store_dimension() {
        // The whole pipeline is fixed at EMBED_DIM, so the default must match.
        assert_eq!(HashEmbedder::default().dimensions(), EMBED_DIM);
    }

    #[test]
    fn noop_reranker_keeps_input_order() {
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(NoopReranker.rerank("q", &docs), vec![0, 1, 2]);
        assert!(NoopReranker.rerank("q", &[]).is_empty());
    }

    #[test]
    fn fallbacks_report_their_names() {
        assert!(HashEmbedder::default().name().contains("hash"));
        assert_eq!(NoopReranker.name(), "none");
    }

    #[test]
    fn source_language_recognizes_code_and_rejects_the_rest() {
        assert_eq!(source_language("src/lib.rs"), Some("rust"));
        assert_eq!(source_language("ui/src/App.tsx"), Some("typescript"));
        assert_eq!(source_language("scripts/Build.PY"), Some("python")); // case-insensitive
        assert_eq!(source_language("README.md"), None);
        assert_eq!(source_language("Cargo.lock"), None);
        assert_eq!(source_language("package-lock.json"), None);
        assert_eq!(source_language("assets/logo.png"), None);
        assert_eq!(source_language("LICENSE"), None);
    }

    #[test]
    fn chunk_code_ast_keeps_functions_whole() {
        // Each function stays intact and every function body ends up in some chunk.
        let code = "fn alpha() {\n    let a = 1;\n    println!(\"{a}\");\n}\n\nfn beta() {\n    let b = 2;\n    println!(\"{b}\");\n}\n";
        let chunks = chunk_code(code, "rust");
        assert!(!chunks.is_empty());
        // No chunk splits a function header from its body: a chunk that contains `fn alpha` also
        // contains alpha's body line.
        let joined: String = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("fn alpha"));
        assert!(joined.contains("fn beta"));
        for c in &chunks {
            if c.text.contains("fn alpha") {
                assert!(c.text.contains("println!(\"{a}\")"), "alpha body split off");
            }
        }
        assert!(chunk_code("", "rust").is_empty());
    }

    #[test]
    fn chunk_code_drops_degenerate_fragments() {
        for junk in [
            "{",
            "(",
            "const",
            "mod tests",
            "function()",
            "  \n  ",
            "} else {",
        ] {
            assert!(is_degenerate_code_chunk(junk), "should drop: {junk:?}");
        }
        for keep in [
            "export const VERSION = \"1.0\"",
            "let routes = HashMap::new();",
            "fn parse_manifest() {}",
            "fn new() -> Self",
        ] {
            assert!(!is_degenerate_code_chunk(keep), "should keep: {keep:?}");
        }
    }

    #[test]
    fn chunk_code_merges_small_nodes_no_micro_chunks() {
        // A big `mod tests { ... }` must not split into a bare header, a `{` fragment and slivers.
        let mut src = String::from(
            "use std::collections::HashMap;\n\npub struct Router { routes: HashMap<String, String> }\n\n\
             impl Router {\n    pub fn forge_for_repo(&self, repo_id: &str) -> Option<&String> {\n\
             let forge = repo_id.split('/').next()?;\n        self.routes.get(forge)\n    }\n}\n\n\
             #[cfg(test)]\nmod tests {\n    use super::*;\n",
        );
        for i in 0..20 {
            src.push_str(&format!(
                "    #[test]\n    fn case_{i}() {{\n        let r = Router::new();\n        \
                 assert!(r.forge_for_repo(\"gh/acme/widget\").is_none());\n    }}\n\n"
            ));
        }
        src.push_str("}\n");

        let chunks = chunk_code(&src, "rust");
        assert!(!chunks.is_empty());
        for c in &chunks {
            // Strip the breadcrumb (a leading comment line) before judging the body.
            let body = c.text.strip_prefix("//").map_or(c.text.as_str(), |rest| {
                rest.split_once('\n').map_or("", |(_, b)| b)
            });
            assert!(
                !is_degenerate_code_chunk(body),
                "degenerate chunk leaked: {:?}",
                c.text
            );
        }
        assert!(chunks.iter().any(|c| c.text.contains("forge_for_repo")));
    }

    #[test]
    fn scope_breadcrumb_names_the_enclosing_decls() {
        let src = "impl Widget {\n    fn render(&self) {\n        let x = 1;\n    }\n}\n";
        let language = tree_sitter_language("rust").unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&language).unwrap();
        let tree = parser.parse(src, None).unwrap();
        let off = src.find("let x").unwrap();
        assert_eq!(
            scope_breadcrumb(&tree, src, off).as_deref(),
            Some("Widget > render")
        );
        let top = "const N: u32 = 1;\n";
        let t2 = {
            let mut p = tree_sitter::Parser::new();
            p.set_language(&language).unwrap();
            p.parse(top, None).unwrap()
        };
        assert_eq!(scope_breadcrumb(&t2, top, 0), None);
    }

    #[test]
    fn chunk_code_prepends_the_scope_comment() {
        // A method chunk carries its enclosing impl/fn as a leading comment; the body stays intact.
        let src =
            "impl Widget {\n    fn render(&self) {\n        let layout = compute();\n    }\n}\n";
        let chunks = chunk_code(src, "rust");
        assert!(!chunks.is_empty());
        let body = chunks
            .iter()
            .find(|c| c.text.contains("compute()"))
            .unwrap();
        assert!(
            body.text.starts_with("// Widget"),
            "chunk should start with the scope breadcrumb comment, got: {:?}",
            body.text
        );
        assert!(body.text.contains("fn render"), "body must stay intact");
    }

    #[test]
    fn chunk_code_uses_hash_comment_for_python() {
        let src = "class Widget:\n    def render(self):\n        layout = compute()\n";
        let chunks = chunk_code(src, "python");
        let body = chunks
            .iter()
            .find(|c| c.text.contains("compute()"))
            .unwrap();
        assert!(
            body.text.starts_with("# Widget"),
            "python breadcrumb uses a # comment, got: {:?}",
            body.text
        );
    }

    #[test]
    fn chunk_code_falls_back_for_unknown_language() {
        // No grammar for this label -> line-window fallback still produces non-empty chunks.
        let code = "line one\nline two\nline three";
        let chunks = chunk_code(code, "cobol");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.text.contains("line two")));
    }

    #[test]
    fn chunk_markdown_splits_and_drops_blanks() {
        let md = "# Title\n\nA paragraph about authentication.\n\n## Section\n\nMore text here.";
        let chunks = chunk_markdown(md);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.text.trim().is_empty()));
        assert!(chunk_markdown("").is_empty());
    }

    #[test]
    fn cosine_of_mismatched_lengths_is_zero() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
    }

    // Real-model smoke tests. Ignored by default: they download models and need the `fastembed`
    // feature and network. Run manually with:
    //   cargo test -p core-embed --features fastembed -- --ignored
    #[cfg(feature = "fastembed")]
    #[test]
    #[ignore = "downloads the model + needs onnxruntime/network"]
    fn real_model_embeds_and_ranks_related_code() {
        let e = FastEmbedder::new().expect("load jina-embeddings-v2-base-code");
        assert_eq!(e.dimensions(), EMBED_DIM);
        let auth = e.embed("fn authenticate(user: &User) -> bool");
        assert_eq!(auth.len(), EMBED_DIM);
        let related = e.embed("fn login(account: Account) -> Session");
        let unrelated = e.embed("fn render_chart(points: &[Point])");
        assert!(cosine_similarity(&auth, &related) > cosine_similarity(&auth, &unrelated));
    }

    // Loads the bundled model files via the production `from_path`, so it catches a broken
    // bundle. Needs the files present (`just fetch-model`) and onnxruntime. Path is relative to
    // this crate.
    #[cfg(feature = "fastembed")]
    #[test]
    #[ignore = "needs the bundled model files (just fetch-model) + onnxruntime"]
    fn bundled_models_load_from_path() {
        let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../hosts/desktop/resources/models");
        let e = FastEmbedder::from_path(&models.join("jina-embeddings-v2-base-code"))
            .expect("load bundled jina embedding model");
        assert_eq!(e.embed("fn main() {}").len(), EMBED_DIM);
        // Related code must score above unrelated; catches a wrong ONNX graph or mismatched
        // tokenizer.
        let auth = e.embed("fn authenticate(user: &User) -> bool");
        let related = e.embed("fn login(account: Account) -> Session");
        let unrelated = e.embed("fn render_chart(points: &[Point])");
        assert!(
            cosine_similarity(&auth, &related) > cosine_similarity(&auth, &unrelated),
            "bundled from_path embeddings do not capture meaning"
        );
        let r = CrossEncoderReranker::from_path(&models.join("jina-reranker-v1-turbo-en"))
            .expect("load bundled jina reranker");
        let docs = vec!["unrelated".to_string(), "authenticate a user".to_string()];
        assert_eq!(r.rerank("how to authenticate", &docs).first(), Some(&1));
    }

    #[cfg(feature = "fastembed")]
    #[test]
    #[ignore = "downloads the model + needs onnxruntime/network"]
    fn real_reranker_orders_by_relevance() {
        let r = CrossEncoderReranker::new().expect("load jina-reranker-v1-turbo-en");
        let docs = vec![
            "fn render_chart(points: &[Point])".to_string(),
            "fn authenticate(user: &User) -> bool".to_string(),
        ];
        let order = r.rerank("how do we authenticate a user", &docs);
        assert_eq!(order.first(), Some(&1));
    }
}
