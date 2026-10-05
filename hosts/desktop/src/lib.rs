//! The desktop shell's boundary surface: typed commands and events between the UI and the
//! host-agnostic core, plus the tauri-specta builder that generates the TypeScript bindings.
//! Lives in the lib target (no `generate_context!`) so the bindings exporter test runs headless.

use std::sync::Arc;

mod credentials;
mod forge;
mod jira;
mod osv;
pub use credentials::{CredentialStore, KeyringCredentials, MemoryCredentials};
pub use forge::AnyForge;

use core_model::Health;
use core_store::{Board, BoardKind, ForgeRow, Repo, Settings, SourceRow, Store};
use core_sync::{
    repo_id as namespaced_repo_id, FullSyncReport, SyncEngine, SyncPhase, SyncProgress,
};
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::State;
use tauri_specta::{collect_commands, collect_events, Builder, Event};

/// Health/ping command: a typed round trip UI -> shell -> core.
#[tauri::command]
#[specta::specta]
fn health() -> Health {
    core_app::health()
}

/// The sync engine the desktop shell runs, generic over [`AnyForge`] so it routes each
/// repo/board to its configured forge by id. Held in an `Arc` so the scheduler task can share
/// it. The forge set is built from the `forges` table at startup; edits apply on next launch.
pub type DesktopSyncEngine = SyncEngine<AnyForge>;

/// Shared shell state. `store` is always present; `engine` is `None` until some connection is
/// authorized. `embedder` is the engine's indexing embedder, held so `semantic_search` can embed a
/// query without the engine or a token.
pub struct AppState {
    pub store: Store,
    /// The live sync engine, swapped in by [`install_engine`] when the connection set or its
    /// credentials change.
    engine: std::sync::RwLock<Option<Arc<DesktopSyncEngine>>>,
    /// Tasks owned by the current engine (index worker, scheduler). Aborted before a swap.
    engine_tasks: std::sync::Mutex<Vec<tauri::async_runtime::JoinHandle<()>>>,
    pub embedder: Arc<dyn core_embed::Embedder>,
    /// Second-stage reranker over hybrid search hits: cross-encoder under `fastembed`, else no-op.
    pub reranker: Arc<dyn core_embed::Reranker>,
    /// Forge access tokens, stored in the OS keychain, never in the DB.
    pub credentials: Arc<dyn CredentialStore>,
    /// The database file path, surfaced in the Debug view.
    pub db_path: String,
    /// Writable directory for downloaded LLM models (OS app-data dir, next to `orgonzola.db`).
    /// Kept out of the source tree so a download does not trigger the dev file-watcher.
    pub llm_download_dir: std::path::PathBuf,
    /// Directory the bundled default model ships in (`just run` env dir in dev, else the app
    /// resource dir). Searched after the download dir.
    pub llm_bundled_dir: Option<std::path::PathBuf>,
    /// Bundled ONNX embedding model directory, kept so the storage report can size it.
    pub embed_model_dir: Option<std::path::PathBuf>,
    /// Bundled ONNX reranker model directory, for the same reason.
    pub rerank_model_dir: Option<std::path::PathBuf>,
}

impl AppState {
    /// Assemble the shell state with no engine yet; [`install_engine`] builds one.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Store,
        embedder: Arc<dyn core_embed::Embedder>,
        reranker: Arc<dyn core_embed::Reranker>,
        credentials: Arc<dyn CredentialStore>,
        db_path: String,
        llm_download_dir: std::path::PathBuf,
        llm_bundled_dir: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            store,
            engine: std::sync::RwLock::new(None),
            engine_tasks: std::sync::Mutex::new(Vec::new()),
            embedder,
            reranker,
            credentials,
            db_path,
            llm_download_dir,
            llm_bundled_dir,
            embed_model_dir: None,
            rerank_model_dir: None,
        }
    }

    /// Record where the bundled ONNX models live, for the storage report. Only the shell knows
    /// these resolved resource paths, so `new` does not take them.
    pub fn with_model_dirs(
        mut self,
        embed: Option<std::path::PathBuf>,
        rerank: Option<std::path::PathBuf>,
    ) -> Self {
        self.embed_model_dir = embed;
        self.rerank_model_dir = rerank;
        self
    }

    /// The current engine, if any. Returns a clone so no caller holds the lock across an await.
    pub fn engine(&self) -> Option<Arc<DesktopSyncEngine>> {
        self.engine.read().ok().and_then(|e| e.clone())
    }

    /// Swap in `engine` and its tasks, aborting the previous engine's tasks first. Returns whether
    /// an engine is now installed.
    fn swap_engine(
        &self,
        engine: Option<Arc<DesktopSyncEngine>>,
        tasks: Vec<tauri::async_runtime::JoinHandle<()>>,
    ) -> bool {
        let installed = engine.is_some();
        if let Ok(mut slot) = self.engine.write() {
            *slot = engine;
        }
        if let Ok(mut running) = self.engine_tasks.lock() {
            for handle in running.drain(..) {
                handle.abort();
            }
            *running = tasks;
        }
        installed
    }

    /// The directory that holds `file`: the download dir first, then the bundled dir. `None` if
    /// neither has it.
    fn llm_dir_for(&self, file: &str) -> Option<String> {
        [Some(&self.llm_download_dir), self.llm_bundled_dir.as_ref()]
            .into_iter()
            .flatten()
            .find(|d| d.join(file).is_file())
            .map(|d| d.to_string_lossy().into_owned())
    }
}

/// Open the shell's store at `db_path`. `None` if it cannot be opened.
pub async fn build_store(db_path: &str) -> Option<Store> {
    Store::open(db_path).await.ok()
}

/// Build the sync engine from the `forges` table over a shared `store`, with `embedder` as its
/// indexing embedder. Seeds a default GitHub forge from `GITHUB_TOKEN` if the table is empty.
/// Returns `None` when no forge is configured.
pub async fn build_engine(
    store: Store,
    embedder: Arc<dyn core_embed::Embedder>,
    credentials: &dyn CredentialStore,
) -> Option<DesktopSyncEngine> {
    seed_env_forge(&store, credentials).await;
    let forges = store.forges().await.ok()?;
    let mut engine = SyncEngine::empty(store).with_embedder(embedder);
    let mut any_built = false;
    for f in forges {
        // A forge with no stored credential is skipped, so its board shows "not connected".
        let Some(token) = credentials.get(&f.id) else {
            eprintln!(
                "forge {} has no stored credential; skipping (not connected)",
                f.id
            );
            continue;
        };
        match AnyForge::from_parts(&f.kind, &f.base_url, &token) {
            Some(any) => {
                engine = engine.with_forge(f.id, any);
                any_built = true;
            }
            None => eprintln!(
                "skipping forge {} ({}): unknown kind or bad base url",
                f.id, f.kind
            ),
        }
    }
    any_built.then_some(engine)
}

/// Build the engine from the current connection config and keychain and install it, replacing any
/// previous one. The only place the engine is created; `main.rs` calls it at startup and every
/// connection or credential change calls it again. Returns whether an engine is now installed.
/// Respawns the index worker and the scheduler with the engine; the old tasks are aborted in
/// [`AppState::swap_engine`]. Aborting the scheduler mid-pass is safe: sync is cursor-based and
/// resumes from the last committed watermark.
pub async fn install_engine(app: &tauri::AppHandle) -> bool {
    use tauri::Manager;
    let state = app.state::<AppState>();
    let (store, embedder, credentials) = (
        state.store.clone(),
        state.embedder.clone(),
        state.credentials.clone(),
    );
    let built = build_engine(store.clone(), embedder, credentials.as_ref()).await;
    let mut tasks = Vec::new();
    let engine = built.map(|e| {
        let (index_tx, index_rx) = tokio::sync::mpsc::channel(256);
        let engine = Arc::new(e.with_index_queue(index_tx));

        let worker = engine.clone();
        let handle = app.clone();
        tasks.push(tauri::async_runtime::spawn(async move {
            worker
                .run_index_worker(index_rx, move |p| {
                    let _ = IndexProgressEvent::from(p).emit(&handle);
                })
                .await;
        }));

        let scheduler = engine.clone();
        let handle = app.clone();
        let snap_store = store;
        tasks.push(tauri::async_runtime::spawn(async move {
            scheduler
                .run_scheduler_over_registry(move |progress| {
                    // Record a metric snapshot for the polled repo.
                    if let Some(tick) = &progress.finished {
                        if tick.report.is_ok() {
                            let store = snap_store.clone();
                            let repo_id = tick.source_id.clone();
                            tauri::async_runtime::spawn(async move {
                                record_repo_snapshot(&store, &repo_id).await;
                            });
                        }
                    }
                    let _ = SyncProgressEvent::from(progress).emit(&handle);
                })
                .await;
        }));

        engine
    });
    state.swap_engine(engine, tasks)
}

/// Seed a default GitHub forge from `GITHUB_TOKEN` when no forge is configured, so an env-only
/// install keeps syncing. Config goes in the DB, the token in the keychain. Idempotent. Also
/// points unassigned boards at the seeded forge.
async fn seed_env_forge(store: &Store, credentials: &dyn CredentialStore) {
    let Ok(forges) = store.forges().await else {
        return;
    };
    if !forges.is_empty() {
        return;
    }
    let Some(token) = std::env::var("GITHUB_TOKEN").ok().filter(|s| !s.is_empty()) else {
        return;
    };
    let forge = ForgeRow {
        id: "github".to_string(),
        name: "GitHub".to_string(),
        kind: "github".to_string(),
        base_url: "https://api.github.com".to_string(),
        oauth_client_id: None,
    };
    if let Err(e) = store.upsert_forge(&forge).await {
        eprintln!("could not seed the default GitHub forge: {e}");
        return;
    }
    if let Err(e) = credentials.set(&forge.id, &token) {
        eprintln!("could not store the seeded GitHub token in the keychain: {e}");
    }
    eprintln!("seeded a default GitHub forge from GITHUB_TOKEN");
    // Point forge-less boards at the seed.
    if let Ok(boards) = store.boards().await {
        for b in boards.iter().filter(|b| b.forge_id.is_none()) {
            let _ = store.set_board_forge(&b.id, Some(&forge.id)).await;
        }
    }
}

/// Choose the indexing embedder: under `fastembed`, the bundled model in `model_dir`, falling back
/// to `HashEmbedder` if the directory is missing or unreadable. Without the feature it is always
/// `HashEmbedder`. The model is never downloaded at runtime.
#[cfg(feature = "fastembed")]
pub fn build_embedder(model_dir: Option<std::path::PathBuf>) -> Arc<dyn core_embed::Embedder> {
    if let Some(dir) = model_dir {
        match core_embed::FastEmbedder::from_path(&dir) {
            Ok(model) => return Arc::new(model),
            Err(e) => eprintln!("fastembed: {e}; falling back to the deterministic embedder"),
        }
    } else {
        eprintln!(
            "fastembed: no bundled model directory; falling back to the deterministic embedder"
        );
    }
    Arc::new(core_embed::HashEmbedder::default())
}

/// Without `fastembed`, indexing uses `HashEmbedder`.
#[cfg(not(feature = "fastembed"))]
pub fn build_embedder(_model_dir: Option<std::path::PathBuf>) -> Arc<dyn core_embed::Embedder> {
    Arc::new(core_embed::HashEmbedder::default())
}

/// Choose the reranker: the real cross-encoder from the bundled `model_dir` under `fastembed`,
/// falling back to the no-op reranker if the directory is missing or unreadable.
#[cfg(feature = "fastembed")]
pub fn build_reranker(model_dir: Option<std::path::PathBuf>) -> Arc<dyn core_embed::Reranker> {
    if let Some(dir) = model_dir {
        match core_embed::CrossEncoderReranker::from_path(&dir) {
            Ok(model) => return Arc::new(model),
            Err(e) => eprintln!("reranker: {e}; falling back to the no-op reranker"),
        }
    } else {
        eprintln!("reranker: no bundled model directory; falling back to the no-op reranker");
    }
    Arc::new(core_embed::NoopReranker)
}

/// Without `fastembed`, search uses the no-op reranker.
#[cfg(not(feature = "fastembed"))]
pub fn build_reranker(_model_dir: Option<std::path::PathBuf>) -> Arc<dyn core_embed::Reranker> {
    Arc::new(core_embed::NoopReranker)
}

/// What a `sync_repo` run persisted: one count per activity type.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SyncSummary {
    pub commits: u32,
    pub pull_requests: u32,
    pub issues: u32,
    pub releases: u32,
    pub reviews: u32,
    pub ci_runs: u32,
    pub dependencies: u32,
    pub code_files: u32,
}

impl From<FullSyncReport> for SyncSummary {
    fn from(r: FullSyncReport) -> Self {
        Self {
            commits: r.commits as u32,
            pull_requests: r.pull_requests as u32,
            issues: r.issues as u32,
            releases: r.releases as u32,
            reviews: r.reviews as u32,
            ci_runs: r.ci_runs as u32,
            dependencies: r.dependencies as u32,
            code_files: r.code_files as u32,
        }
    }
}

/// A configured watch source. Boundary form of `core_store::SourceRow`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SourceView {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub ownership: String,
    pub filters: Vec<String>,
    pub stale_pr_days: u32,
}

impl From<SourceRow> for SourceView {
    fn from(s: SourceRow) -> Self {
        Self {
            id: s.id,
            kind: s.kind,
            name: s.name,
            ownership: s.ownership,
            filters: s.filters,
            stale_pr_days: s.stale_pr_days,
        }
    }
}

/// A configured forge: non-secret config plus `connected`. The token never crosses the boundary.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ForgeView {
    pub id: String,
    pub name: String,
    /// "github" | "gitea".
    pub kind: String,
    pub base_url: String,
    /// The GitHub OAuth app client id for the device flow (public). `None` = PAT-only.
    pub oauth_client_id: Option<String>,
    /// Whether a credential is stored for this forge in the OS keychain.
    pub connected: bool,
}

/// Build a `ForgeView` from a row and the keychain connection status.
fn forge_view(row: ForgeRow, credentials: &dyn CredentialStore) -> ForgeView {
    let connected = credentials.has(&row.id);
    ForgeView {
        id: row.id,
        name: row.name,
        kind: row.kind,
        base_url: row.base_url,
        oauth_client_id: row.oauth_client_id,
        connected,
    }
}

/// A started device-flow authorization, pushed to the UI to show `user_code` and open
/// `verification_uri`.
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
pub struct DeviceAuthEvent {
    pub forge_id: String,
    pub user_code: String,
    pub verification_uri: String,
}

/// Forge kinds the host can build. Keep in sync with `AnyForge::from_parts`.
const FORGE_KINDS: [&str; 2] = ["github", "gitea"];

/// Slugify a forge name into a stable id segment (no `/`), deduped against existing ids with
/// `-2`, `-3`, ...
fn forge_id_from_name(name: &str, existing: &[ForgeRow]) -> String {
    let base: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let base = base.trim_matches('-').to_string();
    let base = if base.is_empty() {
        "forge".to_string()
    } else {
        base
    };
    if !existing.iter().any(|f| f.id == base) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|cand| !existing.iter().any(|f| &f.id == cand))
        .unwrap_or(base)
}

/// Slug id from a display name, suffixed `-2`, `-3`, ... to avoid `existing` ids.
fn unique_id_from_name<'a>(name: &str, existing: impl Iterator<Item = &'a str> + Clone) -> String {
    let base: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let base = base.trim_matches('-').to_string();
    let base = if base.is_empty() {
        "item".to_string()
    } else {
        base
    };
    if !existing.clone().any(|id| id == base) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|cand| !existing.clone().any(|id| id == cand))
        .unwrap_or(base)
}

/// Default API base URL for a kind. Only GitHub has one; self-hosted Gitea and Jira sites
/// cannot be guessed and stay required.
fn default_base_url(kind: &str) -> Option<&'static str> {
    match kind {
        "github" => Some("https://api.github.com"),
        _ => None,
    }
}

/// Why a base URL is required, worded for the chosen kind.
fn missing_base_url_error(kind: &str) -> String {
    match kind {
        "gitea" => "a Gitea / Forgejo connection needs its server URL, for example \
https://git.example.com/api/v1"
            .into(),
        "jira" => {
            "a Jira connection needs its site URL, for example https://acme.atlassian.net".into()
        }
        _ => format!("a {kind} connection needs its base URL"),
    }
}

/// Host part of a base URL: no scheme, userinfo, port or path. `None` if no http(s) authority.
fn url_host(base_url: &str) -> Option<String> {
    let url = base_url.trim();
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest.split('/').next()?;
    let host = authority.rsplit('@').next()?.split(':').next()?;
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

/// Display name for an unnamed connection: brand for public services, host for self-hosted ones.
fn default_connection_name(kind: &str, base_url: &str) -> String {
    match (kind, url_host(base_url).as_deref()) {
        ("github", None | Some("github.com") | Some("api.github.com")) => "GitHub".to_string(),
        ("gitea", None) => "Gitea".to_string(),
        ("jira", None) => "Jira".to_string(),
        (_, Some(host)) => host.to_string(),
        (kind, None) => kind.to_string(),
    }
}

/// Keep a derived name distinct from those in use, by suffixing " (2)", " (3)", ...
fn unique_name<'a>(base: &str, existing: impl Iterator<Item = &'a str> + Clone) -> String {
    if !existing.clone().any(|n| n == base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base} ({n})"))
        .find(|cand| !existing.clone().any(|n| n == cand))
        .unwrap_or_else(|| base.to_string())
}

/// Fill in what the user left blank on add/edit: the kind's default endpoint and a derived display
/// name. Inputs are already trimmed. Errors when there is no default endpoint and no URL.
fn resolve_connection_fields(
    name: &str,
    kind: &str,
    base_url: &str,
    existing_names: &[String],
) -> Result<(String, String), String> {
    let base_url = if base_url.is_empty() {
        default_base_url(kind)
            .ok_or_else(|| missing_base_url_error(kind))?
            .to_string()
    } else {
        base_url.to_string()
    };
    let name = if name.is_empty() {
        unique_name(
            &default_connection_name(kind, &base_url),
            existing_names.iter().map(String::as_str),
        )
    } else {
        name.to_string()
    };
    Ok((name, base_url))
}

/// Message shown when no forge is authorized yet. One string shared by the sync path and pickers.
const NO_CONNECTION: &str =
    "no connection is authorized yet - add one under Settings > Connections and connect it";

/// Forge kinds the host can build, checked before deriving a default from the kind.
fn validate_forge_kind(kind: &str) -> Result<(), String> {
    if !FORGE_KINDS.contains(&kind) {
        return Err("connection kind must be \"github\" or \"gitea\"".into());
    }
    Ok(())
}

fn validate_forge_fields(name: &str, kind: &str, base_url: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("forge name cannot be empty".into());
    }
    validate_forge_kind(kind)?;
    if base_url.is_empty() {
        return Err("base URL cannot be empty".into());
    }
    // Require an explicit http(s) scheme so the authenticated client cannot be pointed at an
    // unexpected target.
    if !base_url.starts_with("https://") && !base_url.starts_with("http://") {
        return Err("base URL must start with http:// or https://".into());
    }
    Ok(())
}

fn clean_opt(s: Option<String>) -> Option<String> {
    s.map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// Every configured forge with its connection status (never its token).
#[tauri::command]
#[specta::specta]
async fn list_forges(state: State<'_, AppState>) -> Result<Vec<ForgeView>, String> {
    let forges = state.store.forges().await.map_err(|e| e.to_string())?;
    Ok(forges
        .into_iter()
        .map(|f| forge_view(f, state.credentials.as_ref()))
        .collect())
}

/// Add a forge (kind `github` | `gitea`) with its API base URL and optional OAuth client id. The
/// user connects it afterwards, which stores the token in the keychain.
#[tauri::command]
#[specta::specta]
async fn add_forge(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    name: String,
    kind: String,
    base_url: String,
    oauth_client_id: Option<String>,
) -> Result<ForgeView, String> {
    let name = name.trim();
    let base_url = base_url.trim().trim_end_matches('/');
    validate_forge_kind(&kind)?;
    let existing = state.store.forges().await.map_err(|e| e.to_string())?;
    let taken: Vec<String> = existing.iter().map(|f| f.name.clone()).collect();
    // A blank name or URL means the default for this kind.
    let (name, base_url) = resolve_connection_fields(name, &kind, base_url, &taken)?;
    validate_forge_fields(&name, &kind, &base_url)?;
    let row = ForgeRow {
        id: forge_id_from_name(&name, &existing),
        name,
        kind,
        base_url,
        oauth_client_id: clean_opt(oauth_client_id),
    };
    state
        .store
        .upsert_forge(&row)
        .await
        .map_err(|e| e.to_string())?;
    // Rebuild the engine so the change takes effect in the running process.
    install_engine(&app).await;
    Ok(forge_view(row, state.credentials.as_ref()))
}

/// Edit a forge's name, kind, base URL or OAuth client id. Credentials are managed separately.
/// Takes effect on the next app launch.
#[tauri::command]
#[specta::specta]
async fn update_forge(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    id: String,
    name: String,
    kind: String,
    base_url: String,
    oauth_client_id: Option<String>,
) -> Result<ForgeView, String> {
    let name = name.trim();
    let base_url = base_url.trim().trim_end_matches('/');
    validate_forge_kind(&kind)?;
    let existing = state.store.forges().await.map_err(|e| e.to_string())?;
    if !existing.iter().any(|f| f.id == id) {
        return Err(format!("no forge {id}"));
    }
    // Clearing the name reverts to the derived name, which must stay distinct from the other
    // connections' names, not this one's own.
    let taken: Vec<String> = existing
        .iter()
        .filter(|f| f.id != id)
        .map(|f| f.name.clone())
        .collect();
    let (name, base_url) = resolve_connection_fields(name, &kind, base_url, &taken)?;
    validate_forge_fields(&name, &kind, &base_url)?;
    let row = ForgeRow {
        id,
        name,
        kind,
        base_url,
        oauth_client_id: clean_opt(oauth_client_id),
    };
    state
        .store
        .upsert_forge(&row)
        .await
        .map_err(|e| e.to_string())?;
    // Rebuild the engine so the change takes effect in the running process.
    install_engine(&app).await;
    Ok(forge_view(row, state.credentials.as_ref()))
}

/// Delete a forge and its stored credential. Fails if any board still points at it.
#[tauri::command]
#[specta::specta]
async fn delete_forge(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    let boards = state.store.boards().await.map_err(|e| e.to_string())?;
    if boards.iter().any(|b| b.forge_id.as_deref() == Some(&id)) {
        return Err("this forge is still used by a board; reassign those boards first".into());
    }
    state.credentials.delete(&id)?;
    state
        .store
        .remove_forge(&id)
        .await
        .map_err(|e| e.to_string())?;
    // Rebuild the engine so the change takes effect in the running process.
    install_engine(&app).await;
    Ok(())
}

/// Connect a forge by storing a pasted personal access token in the keychain. Returns the forge
/// with `connected: true`.
#[tauri::command]
#[specta::specta]
async fn set_forge_pat(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    forge_id: String,
    token: String,
) -> Result<ForgeView, String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("token cannot be empty".into());
    }
    let row = state
        .store
        .forge(&forge_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no forge {forge_id}"))?;
    state.credentials.set(&forge_id, token)?;
    // Rebuild the engine so the change takes effect in the running process.
    install_engine(&app).await;
    Ok(forge_view(row, state.credentials.as_ref()))
}

/// Disconnect a forge: remove its credential from the keychain. Returns the forge with
/// `connected: false`.
#[tauri::command]
#[specta::specta]
async fn disconnect_forge(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    forge_id: String,
) -> Result<ForgeView, String> {
    let row = state
        .store
        .forge(&forge_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no forge {forge_id}"))?;
    state.credentials.delete(&forge_id)?;
    // Rebuild the engine so the change takes effect in the running process.
    install_engine(&app).await;
    Ok(forge_view(row, state.credentials.as_ref()))
}

/// Built-in GitHub OAuth client id, set at build time via `ORGONZOLA_GITHUB_CLIENT_ID`. Empty when
/// unset, in which case a per-forge `oauth_client_id` or a pasted token is needed. Public, not a
/// secret: the device flow uses no client secret.
const DEFAULT_GITHUB_CLIENT_ID: &str = match option_env!("ORGONZOLA_GITHUB_CLIENT_ID") {
    Some(s) => s,
    None => "",
};

/// Client id for a forge's device flow: its own when set, else the built-in default. `None` when
/// neither exists.
fn resolve_github_client_id(forge_client_id: Option<&str>) -> Option<String> {
    if let Some(id) = forge_client_id.filter(|s| !s.is_empty()) {
        return Some(id.to_string());
    }
    (!DEFAULT_GITHUB_CLIENT_ID.is_empty()).then(|| DEFAULT_GITHUB_CLIENT_ID.to_string())
}

/// Whether a built-in GitHub OAuth client id exists, so the UI can offer "Connect with GitHub".
#[tauri::command]
#[specta::specta]
fn github_oauth_available() -> bool {
    !DEFAULT_GITHUB_CLIENT_ID.is_empty()
}

/// Connect a GitHub forge via the OAuth device flow. Requests a device code, emits a
/// `DeviceAuthEvent` for the UI, then polls until the user authorizes or the code is denied or
/// expires. The token goes to the keychain, never to the DB or the UI. Only valid for a `github`
/// forge with a resolvable OAuth client id.
#[tauri::command]
#[specta::specta]
async fn connect_device_auth(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    forge_id: String,
) -> Result<(), String> {
    let forge = state
        .store
        .forge(&forge_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no forge {forge_id}"))?;
    if forge.kind != "github" {
        return Err(
            "the device flow is GitHub-only; connect this forge with a token instead".into(),
        );
    }
    let client_id = resolve_github_client_id(forge.oauth_client_id.as_deref())
        .ok_or("Connect needs a GitHub OAuth client id. Paste an access token instead, or set one under Advanced.")?;

    let code = core_github::request_device_code(
        core_github::GITHUB_OAUTH_BASE,
        &client_id,
        "repo read:org",
    )
    .await
    .map_err(|e| e.to_string())?;
    let _ = DeviceAuthEvent {
        forge_id: forge_id.clone(),
        user_code: code.user_code.clone(),
        verification_uri: code.verification_uri.clone(),
    }
    .emit(&app);

    let mut interval = code.interval_secs.max(1);
    let mut remaining = code.expires_in_secs;
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        remaining = remaining.saturating_sub(interval);
        match core_github::poll_device_token(
            core_github::GITHUB_OAUTH_BASE,
            &client_id,
            &code.device_code,
        )
        .await
        .map_err(|e| e.to_string())?
        {
            core_github::DevicePoll::Authorized(token) => {
                state.credentials.set(&forge_id, &token)?;
                // Same as the PAT path: the running process must pick up the new credential.
                install_engine(&app).await;
                return Ok(());
            }
            core_github::DevicePoll::Pending => {}
            core_github::DevicePoll::SlowDown => interval += 5,
            core_github::DevicePoll::Denied => return Err("authorization was denied".into()),
            core_github::DevicePoll::Expired => {
                return Err("the device code expired; try connecting again".into())
            }
        }
        if remaining == 0 {
            return Err("the device code expired; try connecting again".into());
        }
    }
}

/// One attention item for the UI. Boundary form of `core_summary::AttentionItem`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AttentionView {
    pub kind: String,
    /// The typed envelope: the flagged `subject`, who to nudge as `actor`, and the `evidence`. Its
    /// identity and outbound links; passed through from `core_model`.
    pub entities: Vec<core_model::LinkedEntity>,
    pub summary: String,
    /// Which repo this item is on. Filled by the caller; `None` if not set.
    pub repo_full_name: Option<String>,
    /// When the flagged thing happened, so the Attention tab can scope to a recent window.
    pub ts: Option<String>,
    /// The deterministic mechanical next step. Empty when there is none.
    pub action: String,
    /// For `aging_wip` items: band of the board's in-progress-time distribution.
    /// "over_p50" | "over_p75" | "over_p90", or null with too little history.
    pub wip_percentile: Option<String>,
    /// What `wip_percentile` is measured over: span, window and sample size. Shown on the badge.
    pub wip_percentile_basis: Option<String>,
}

impl From<core_summary::AttentionItem> for AttentionView {
    fn from(i: core_summary::AttentionItem) -> Self {
        Self {
            kind: i.kind,
            entities: i.entities,
            summary: i.summary,
            repo_full_name: None,
            ts: i.ts,
            action: i.action,
            wip_percentile: i.wip_percentile,
            wip_percentile_basis: i.wip_percentile_basis,
        }
    }
}

/// The current time as RFC-3339, for staleness checks. The host owns the clock; core is
/// clock-free. Public so the scheduler can date metric snapshots.
pub fn now_rfc3339() -> String {
    use time::format_description::well_known::Rfc3339;
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

/// `days` ago from now, as RFC-3339. Bounds ritual windows.
fn days_ago_rfc3339(days: i64) -> String {
    use time::format_description::well_known::Rfc3339;
    (time::OffsetDateTime::now_utc() - time::Duration::days(days))
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

/// What needs attention in a repo (`owner/name`): stale PRs, merged-without-review, failing CI,
/// over the synced store. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn attention(
    state: State<'_, AppState>,
    full_name: String,
) -> Result<Vec<AttentionView>, String> {
    let repo_id = format!("repo:{full_name}");
    let now = now_rfc3339();
    // Same aging bands as the board view, computed from this repo's own completions.
    let repo_ids = vec![repo_id.clone()];
    let bands = core_summary::board_wip_aging_bands(&state.store, &repo_ids, &now)
        .await
        .map_err(|e| e.to_string())?;
    let items = core_summary::attention_enriched(&state.store, &repo_id, &now, &bands)
        .await
        .map_err(|e| e.to_string())?;
    Ok(items.into_iter().map(AttentionView::from).collect())
}

/// A repo's at-a-glance digest. Boundary form of `core_summary::Digest`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DigestView {
    pub generated_at: String,
    pub median_cycle_time_secs: Option<i64>,
    /// Cycle-time phase medians: pickup = created -> first review, review = first review -> merge.
    pub median_pickup_secs: Option<i64>,
    pub median_review_secs: Option<i64>,
    pub wip: u32,
    pub stale_open_prs: u32,
    pub merged_without_review: u32,
    /// Open, unmerged PRs waiting for a first review past the review-wait threshold.
    pub review_wait: u32,
    pub review_concentration: f64,
    pub prose: String,
}

impl From<core_summary::Digest> for DigestView {
    fn from(d: core_summary::Digest) -> Self {
        Self {
            generated_at: d.generated_at,
            median_cycle_time_secs: d.median_cycle_time_secs,
            median_pickup_secs: d.median_pickup_secs,
            median_review_secs: d.median_review_secs,
            wip: d.wip as u32,
            stale_open_prs: d.stale_open_prs as u32,
            merged_without_review: d.merged_without_review as u32,
            review_wait: d.review_wait as u32,
            review_concentration: d.review_concentration,
            prose: d.prose,
        }
    }
}

/// A repo's digest (`owner/name`): flow metrics, watchdog counts and a prose line. Read-only.
#[tauri::command]
#[specta::specta]
async fn digest(state: State<'_, AppState>, full_name: String) -> Result<DigestView, String> {
    let repo_id = format!("repo:{full_name}");
    let d = core_summary::repo_digest(&state.store, &repo_id, &now_rfc3339())
        .await
        .map_err(|e| e.to_string())?;
    Ok(d.into())
}

/// An upstream-risk alert for the UI: a depended-on repo and how many attention items it has.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct UpstreamAlertView {
    pub repo_id: String,
    pub attention_count: u32,
}

impl From<core_summary::UpstreamAlert> for UpstreamAlertView {
    fn from(a: core_summary::UpstreamAlert) -> Self {
        Self {
            repo_id: a.repo_id,
            attention_count: a.attention_count as u32,
        }
    }
}

/// A pull request in the list view: enough to pick one to inspect.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PullRequestView {
    pub id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
}

impl From<core_store::PullRequest> for PullRequestView {
    fn from(p: core_store::PullRequest) -> Self {
        Self {
            id: p.id,
            number: p.number,
            title: p.title,
            state: p.state,
            author_login: p.author_login,
        }
    }
}

/// The pull requests for a repo (`owner/name`), newest first, for the change picker. Read-only.
#[tauri::command]
#[specta::specta]
async fn pull_requests(
    state: State<'_, AppState>,
    full_name: String,
) -> Result<Vec<PullRequestView>, String> {
    let repo_id = format!("repo:{full_name}");
    let prs = state
        .store
        .pull_requests(&repo_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(prs.into_iter().map(PullRequestView::from).collect())
}

/// A linked issue on a change digest. Boundary form of `core_summary::LinkedIssue`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct LinkedIssueView {
    pub id: String,
    pub number: i64,
    pub title: String,
    pub relation: String,
}

impl From<core_summary::LinkedIssue> for LinkedIssueView {
    fn from(l: core_summary::LinkedIssue) -> Self {
        Self {
            id: l.id,
            number: l.number,
            title: l.title,
            relation: l.relation,
        }
    }
}

/// A file a change touched. Boundary form of `core_summary::ChangedFile`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ChangedFileView {
    pub filename: String,
    pub status: String,
    pub additions: i64,
    pub deletions: i64,
    pub patch: Option<String>,
}

impl From<core_summary::ChangedFile> for ChangedFileView {
    fn from(f: core_summary::ChangedFile) -> Self {
        Self {
            filename: f.filename,
            status: f.status,
            additions: f.additions,
            deletions: f.deletions,
            patch: f.patch,
        }
    }
}

/// A per-change digest. Boundary form of `core_summary::ChangeDigest`. `body` is the PR's
/// Markdown description, rendered in the UI; `files` is the diff.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ChangeDigestView {
    pub pr_id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
    pub body: Option<String>,
    pub review_count: u32,
    pub approved: bool,
    pub linked_issues: Vec<LinkedIssueView>,
    /// Jira tickets linked to this PR via a `PROJ-123` key. Filled by the command.
    pub linked_tickets: Vec<JiraTicketView>,
    pub files: Vec<ChangedFileView>,
    pub additions: i64,
    pub deletions: i64,
    pub prose: String,
}

impl From<core_summary::ChangeDigest> for ChangeDigestView {
    fn from(d: core_summary::ChangeDigest) -> Self {
        Self {
            pr_id: d.pr_id,
            number: d.number,
            title: d.title,
            state: d.state,
            author_login: d.author_login,
            body: d.body,
            review_count: d.review_count as u32,
            approved: d.approved,
            linked_issues: d
                .linked_issues
                .into_iter()
                .map(LinkedIssueView::from)
                .collect(),
            linked_tickets: Vec::new(),
            files: d.files.into_iter().map(ChangedFileView::from).collect(),
            additions: d.additions,
            deletions: d.deletions,
            prose: d.prose,
        }
    }
}

/// The change digest for one pull request (by id). `null` if there is no such PR. Read-only.
#[tauri::command]
#[specta::specta]
async fn change_digest(
    state: State<'_, AppState>,
    pr_id: String,
) -> Result<Option<ChangeDigestView>, String> {
    let digest = core_summary::change_digest(&state.store, &pr_id)
        .await
        .map_err(|e| e.to_string())?;
    let Some(digest) = digest else {
        return Ok(None);
    };
    let mut view = ChangeDigestView::from(digest);
    view.linked_tickets = state
        .store
        .jira_issues_for_pr(&pr_id)
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(JiraTicketView::from)
        .collect();
    Ok(Some(view))
}

// ---- Jira tracker ----

/// A registered tracker (Jira site). `connected` reflects only a stored token: a public site needs
/// none, and the two cannot be told apart.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct TrackerView {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub base_url: String,
    pub email: Option<String>,
    pub connected: bool,
}

/// A Jira ticket. Boundary form of `core_store::JiraIssueRow`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct JiraTicketView {
    pub key: String,
    pub title: String,
    pub issue_type: String,
    pub status: String,
    pub status_category: Option<String>,
    pub assignee: Option<String>,
    pub url: Option<String>,
    pub updated_at: Option<String>,
    pub resolved_at: Option<String>,
}

impl From<core_store::JiraIssueRow> for JiraTicketView {
    fn from(j: core_store::JiraIssueRow) -> Self {
        Self {
            key: j.issue_key,
            title: j.title,
            issue_type: j.issue_type,
            status: j.status,
            status_category: j.status_category,
            assignee: j.assignee,
            url: j.url,
            updated_at: j.updated_at,
            resolved_at: j.resolved_at,
        }
    }
}

/// A Jira sprint for the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct JiraSprintView {
    pub name: String,
    pub state: Option<String>,
    pub starts_on: Option<String>,
    pub ends_on: Option<String>,
}

/// A board's Jira surface: its watched projects' tickets and sprints.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardJiraView {
    pub tickets: Vec<JiraTicketView>,
    pub sprints: Vec<JiraSprintView>,
    /// The board's linked projects (empty = no Jira on this board).
    pub projects: Vec<BoardTrackerView>,
}

/// A board's link to a tracker project, with the tracker's display name.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardTrackerView {
    pub tracker_id: String,
    pub tracker_name: String,
    pub project_key: String,
}

fn tracker_view(t: core_store::Tracker, credentials: &dyn CredentialStore) -> TrackerView {
    let connected = credentials.has(&t.id);
    TrackerView {
        id: t.id,
        name: t.name,
        kind: t.kind,
        base_url: t.base_url,
        email: t.email,
        connected,
    }
}

/// Every registered tracker (Jira site). Read-only.
#[tauri::command]
#[specta::specta]
async fn list_trackers(state: State<'_, AppState>) -> Result<Vec<TrackerView>, String> {
    let rows = state.store.trackers().await.map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|t| tracker_view(t, state.credentials.as_ref()))
        .collect())
}

/// Register a Jira site (URL + optional Cloud email). The API token is set separately in the
/// keychain; a public instance needs none.
#[tauri::command]
#[specta::specta]
async fn add_tracker(
    state: State<'_, AppState>,
    name: String,
    base_url: String,
    email: Option<String>,
) -> Result<TrackerView, String> {
    let name = name.trim();
    let base_url = base_url.trim().trim_end_matches('/');
    let existing = state.store.trackers().await.map_err(|e| e.to_string())?;
    let taken: Vec<String> = existing.iter().map(|t| t.name.clone()).collect();
    // Same as a forge: a blank name is derived from the site host.
    let (name, base_url) = resolve_connection_fields(name, "jira", base_url, &taken)?;
    if !base_url.starts_with("https://") && !base_url.starts_with("http://") {
        return Err("base url must start with http(s)://".into());
    }
    let id = unique_id_from_name(&name, existing.iter().map(|t| t.id.as_str()));
    let row = core_store::Tracker {
        id,
        name,
        kind: "jira".to_string(),
        base_url,
        email: email
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty()),
    };
    state
        .store
        .upsert_tracker(&row)
        .await
        .map_err(|e| e.to_string())?;
    Ok(tracker_view(row, state.credentials.as_ref()))
}

/// Store a Jira API token in the keychain for a tracker. An empty token clears it.
#[tauri::command]
#[specta::specta]
async fn set_tracker_token(
    state: State<'_, AppState>,
    tracker_id: String,
    token: String,
) -> Result<TrackerView, String> {
    let row = state
        .store
        .tracker(&tracker_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no tracker {tracker_id}"))?;
    let token = token.trim();
    if token.is_empty() {
        state.credentials.delete(&tracker_id)?;
    } else {
        state.credentials.set(&tracker_id, token)?;
    }
    Ok(tracker_view(row, state.credentials.as_ref()))
}

/// Delete a tracker, its board links and its stored token.
#[tauri::command]
#[specta::specta]
async fn delete_tracker(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let _ = state.credentials.delete(&id);
    state
        .store
        .delete_tracker(&id)
        .await
        .map_err(|e| e.to_string())
}

/// Link a Jira project (by key) to a board. Returns the board's links.
#[tauri::command]
#[specta::specta]
async fn link_board_tracker(
    state: State<'_, AppState>,
    board_id: String,
    tracker_id: String,
    project_key: String,
) -> Result<Vec<BoardTrackerView>, String> {
    let project_key = project_key.trim().to_uppercase();
    if project_key.is_empty() {
        return Err("project key cannot be empty".into());
    }
    state
        .store
        .link_board_tracker(&board_id, &tracker_id, &project_key)
        .await
        .map_err(|e| e.to_string())?;
    board_tracker_views(&state.store, &board_id).await
}

/// Unlink a Jira project from a board. Returns the remaining links.
#[tauri::command]
#[specta::specta]
async fn unlink_board_tracker(
    state: State<'_, AppState>,
    board_id: String,
    tracker_id: String,
    project_key: String,
) -> Result<Vec<BoardTrackerView>, String> {
    state
        .store
        .unlink_board_tracker(&board_id, &tracker_id, &project_key)
        .await
        .map_err(|e| e.to_string())?;
    board_tracker_views(&state.store, &board_id).await
}

async fn board_tracker_views(
    store: &Store,
    board_id: &str,
) -> Result<Vec<BoardTrackerView>, String> {
    let trackers = store.trackers().await.map_err(|e| e.to_string())?;
    let name_of = |id: &str| {
        trackers
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.name.clone())
            .unwrap_or_else(|| id.to_string())
    };
    Ok(store
        .board_trackers(board_id)
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|bt| BoardTrackerView {
            tracker_name: name_of(&bt.tracker_id),
            tracker_id: bt.tracker_id,
            project_key: bt.project_key,
        })
        .collect())
}

/// The board's Jira tickets, sprints and project links. Read-only.
#[tauri::command]
#[specta::specta]
async fn board_jira(state: State<'_, AppState>, id: String) -> Result<BoardJiraView, String> {
    let tickets = state
        .store
        .jira_issues_for_board(&id)
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(JiraTicketView::from)
        .collect();
    let sprints = state
        .store
        .jira_sprints_for_board(&id)
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|s| JiraSprintView {
            name: s.name,
            state: s.state,
            starts_on: s.starts_on,
            ends_on: s.ends_on,
        })
        .collect();
    let projects = board_tracker_views(&state.store, &id).await?;
    Ok(BoardJiraView {
        tickets,
        sprints,
        projects,
    })
}

/// One watched repo's home-view state: its digest, attention items and upstream risk.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RepoOverview {
    pub repo_id: String,
    pub full_name: String,
    pub ownership: String,
    pub digest: DigestView,
    pub attention: Vec<AttentionView>,
    /// True count of this repo's attention items before the board-wide payload bound. Can exceed
    /// `attention.len()` on a big board, since `board_overview` keeps only the most severe items.
    pub attention_total: u32,
    /// Merged PRs that could not be judged for review coverage because they merged before the
    /// earliest review synced for the repo. Reviews are fetched only for PRs changed since, so
    /// such a PR was never checked. Not counted in `attention`.
    pub merged_without_review_unjudged: u32,
    /// Merged PRs that could not be judged for a tracked work item because the repo has no
    /// issue-link data synced. Same idea as above, for the `orphan_pr` signal.
    pub orphan_pr_unjudged: u32,
    pub upstream: Vec<UpstreamAlertView>,
    /// Whether this repo's source is indexed for code search. `false` skips code fetch and embed.
    pub index_code: bool,
    /// Whether this repo is a fork. The UI folds observed forks out of the flagged list.
    pub is_fork: bool,
    /// This fork's upstream `owner/name`, so the UI can note "fork of X".
    pub parent_full_name: Option<String>,
}

/// A declared dependency. Boundary form of `core_store::DependencyRow`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DependencyView {
    pub ecosystem: String,
    pub name: String,
    pub version_req: Option<String>,
    pub kind: String,
}

impl From<core_store::DependencyRow> for DependencyView {
    fn from(d: core_store::DependencyRow) -> Self {
        Self {
            ecosystem: d.ecosystem,
            name: d.name,
            version_req: d.version_req,
            kind: d.kind,
        }
    }
}

/// One semantic-search hit: the matched chunk and its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SearchHit {
    pub ref_kind: String,
    pub ref_id: String,
    pub chunk: String,
    pub score: f32,
    /// Forge web URL of the source file for a code hit. `None` for activity hits or an unknown
    /// forge web base.
    pub url: Option<String>,
}

impl From<core_store::EmbeddingHit> for SearchHit {
    fn from(h: core_store::EmbeddingHit) -> Self {
        Self {
            ref_kind: h.ref_kind,
            ref_id: h.ref_id,
            chunk: h.chunk,
            score: h.score,
            url: None,
        }
    }
}

/// A forge's web root from its API `base_url`: api.github.com -> github.com; GHE/Gitea drop the
/// `/api/v3` or `/api/v1` suffix; otherwise the base as-is.
fn derive_web_base(base_url: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if b == "https://api.github.com" {
        return "https://github.com".to_string();
    }
    for suffix in ["/api/v3", "/api/v1"] {
        if let Some(stripped) = b.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    b.to_string()
}

/// Forge file URL for a code hit (`ref_kind == "code"`, `ref_id == "<repo_id>#<path>"`), from the
/// per-forge web bases. `None` for non-code hits or an unknown forge.
fn code_hit_url(
    ref_kind: &str,
    ref_id: &str,
    web_bases: &std::collections::HashMap<String, String>,
) -> Option<String> {
    if ref_kind != "code" {
        return None;
    }
    let (repo_id, path) = ref_id.split_once('#')?;
    let base = web_bases.get(core_sync::forge_id_of(repo_id)?)?;
    Some(format!(
        "{base}/{}/blob/HEAD/{path}",
        full_name_of_repo_id(repo_id)
    ))
}

/// Shared retrieval path: hybrid search (vector KNN + FTS5 keyword, fused) then a cross-encoder
/// rerank. `filter` picks the index slice (activity vs code), `scope` the repos. Read-only.
async fn search_and_rerank(
    state: &AppState,
    query: &str,
    k: i64,
    filter: core_store::KindFilter<'_>,
    scope: core_store::RepoScope<'_>,
) -> Result<Vec<SearchHit>, String> {
    // Real embedder/reranker run synchronous ONNX inference (hundreds of ms); use the blocking
    // pool so the async executor is not stalled.
    let vector = {
        let embedder = state.embedder.clone();
        let q = query.to_string();
        tokio::task::spawn_blocking(move || embedder.embed(&q))
            .await
            .map_err(|e| e.to_string())?
    };
    let k = k.clamp(1, 50) as usize;
    let hits = state
        .store
        .hybrid_search(&vector, query, k, filter, scope)
        .await
        .map_err(|e| e.to_string())?;
    // Second stage: the cross-encoder reorders the fused candidates by joint (query, chunk)
    // relevance (no-op without the model). Returns a permutation of the hits' indices.
    let docs: Vec<String> = hits.iter().map(|h| h.chunk.clone()).collect();
    let order = {
        let reranker = state.reranker.clone();
        let q = query.to_string();
        tokio::task::spawn_blocking(move || reranker.rerank(&q, &docs))
            .await
            .map_err(|e| e.to_string())?
    };
    // Per-forge web bases, so a code hit links to the file on its forge.
    let web_bases: std::collections::HashMap<String, String> = state
        .store
        .forges()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|f| (f.id, derive_web_base(&f.base_url)))
        .collect();
    Ok(order
        .into_iter()
        .filter_map(|i| hits.get(i).cloned())
        .map(SearchHit::from)
        .map(|mut h| {
            h.url = code_hit_url(&h.ref_kind, &h.ref_id, &web_bases);
            h
        })
        .collect())
}

/// Backend and store/index status for the Debug view: the active embedder/reranker plus
/// `core_store::StoreCounts`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DebugStatsView {
    /// The active embedder's name (real model or deterministic fallback) and its dimension.
    pub embedder: String,
    pub embedder_dims: u32,
    /// The active reranker's name ("none" for the no-op).
    pub reranker: String,
    /// Whether the `fastembed` feature is built into this binary.
    pub fastembed_built: bool,
    pub db_path: String,
    /// Jobs waiting in the background index queue; 0 if empty or no engine/queue.
    pub index_queue_depth: u32,
    pub repos: u32,
    pub commits: u32,
    pub pull_requests: u32,
    pub issues: u32,
    pub sources: u32,
    pub boards: u32,
    pub embeddings_total: u32,
    pub embeddings_code: u32,
    pub embeddings_activity: u32,
    pub code_files: u32,
    /// Repos currently in the index lane (queued / indexing / errored), for the Debug panel.
    pub index_active: Vec<IndexEntryView>,
    /// Local-LLM narrator status, including why the AI brief falls back when it does.
    pub llm: LlmStatusView,
}

/// The local-LLM narrator's runtime status for the Debug panel.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct LlmStatusView {
    /// Whether the `llm` feature is compiled into this binary.
    pub feature_built: bool,
    /// Whether a model is configured (both env vars set).
    pub configured: bool,
    /// The configured `dir/file`, if any.
    pub model_path: Option<String>,
    /// Whether that GGUF file exists on disk.
    pub model_present: bool,
    /// Whether the model has been loaded this session (the first brief loads it).
    pub loaded: bool,
    /// The most recent engine error (model load / generation), or null if the last run was clean.
    pub last_error: Option<String>,
}

/// One repo's index-lane state, for the Debug "Queues" panel. `done`/`total` carry the embedding
/// sub-progress while `indexing` (0 otherwise).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct IndexEntryView {
    pub repo_id: String,
    pub full_name: String,
    pub state: String,
    pub done: u32,
    pub total: u32,
    /// The failure reason when `state` is "error" (else null).
    pub error: Option<String>,
    /// Files still to index when `state` is "partial"; 0 otherwise.
    pub pending: u32,
    /// Files this pass could not fetch when `state` is "partial"; 0 otherwise.
    pub skipped: u32,
    /// Why a "partial" or "paused" state holds. Separate from `error`: neither is a failure.
    pub note: Option<String>,
}

/// The wire string for an index-lane state.
fn index_state_str(state: core_sync::IndexState) -> &'static str {
    match state {
        core_sync::IndexState::Queued => "queued",
        core_sync::IndexState::Indexing => "indexing",
        core_sync::IndexState::Indexed => "indexed",
        core_sync::IndexState::Partial => "partial",
        core_sync::IndexState::Paused => "paused",
        core_sync::IndexState::Error => "error",
    }
}

impl From<core_sync::IndexProgress> for IndexEntryView {
    fn from(p: core_sync::IndexProgress) -> Self {
        Self {
            repo_id: p.repo_id,
            full_name: p.full_name,
            state: index_state_str(p.state).to_string(),
            done: p.done as u32,
            total: p.total as u32,
            error: p.error,
            pending: p.pending as u32,
            skipped: p.skipped as u32,
            note: p.note,
        }
    }
}

/// Report backend status and current store/index counts. Read-only; works with no token. The
/// embedder/reranker names are what is loaded, not what was requested.
#[tauri::command]
#[specta::specta]
async fn debug_stats(state: State<'_, AppState>) -> Result<DebugStatsView, String> {
    let c = state.store.counts().await.map_err(|e| e.to_string())?;
    let engine = state.engine();
    Ok(DebugStatsView {
        embedder: state.embedder.name().to_string(),
        embedder_dims: state.embedder.dimensions() as u32,
        reranker: state.reranker.name().to_string(),
        fastembed_built: cfg!(feature = "fastembed"),
        db_path: state.db_path.clone(),
        index_queue_depth: engine
            .as_ref()
            .and_then(|e| e.index_queue_depth())
            .unwrap_or(0) as u32,
        repos: c.repos as u32,
        commits: c.commits as u32,
        pull_requests: c.pull_requests as u32,
        issues: c.issues as u32,
        sources: c.sources as u32,
        boards: c.boards as u32,
        embeddings_total: c.embeddings_total as u32,
        embeddings_code: c.embeddings_code as u32,
        embeddings_activity: c.embeddings_activity as u32,
        code_files: c.code_files as u32,
        index_active: engine
            .as_ref()
            .map(|e| e.index_status())
            .unwrap_or_default()
            .into_iter()
            .map(IndexEntryView::from)
            .collect(),
        llm: llm_status(),
    })
}

/// The live index lane alone: the rows of `debug_stats.index_active` without the store-wide
/// counts. The board shell polls this to keep each repo's index badge current.
#[tauri::command]
#[specta::specta]
async fn index_status(state: State<'_, AppState>) -> Result<Vec<IndexEntryView>, String> {
    Ok(state
        .engine()
        .map(|e| e.index_status())
        .unwrap_or_default()
        .into_iter()
        .map(IndexEntryView::from)
        .collect())
}

/// One repo in the store, for the Debug forget list. `pinned` is the user's choice, `discovered`
/// means it came from a board's people; a repo with neither is watched by nothing.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct StoredRepoView {
    pub repo_id: String,
    pub full_name: String,
    pub pinned: bool,
    pub discovered: bool,
}

/// What forgetting a repo removed. Boundary form of `core_store::ForgottenRepo`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ForgottenRepoView {
    pub repo_id: String,
    pub found: bool,
    pub commits: u32,
    pub pull_requests: u32,
    pub issues: u32,
    pub code_files: u32,
    pub index_chunks: u32,
    pub total_rows: u32,
}

/// Every repo in the store, marked with whether a board pins or discovers it. Read-only.
/// Repos nothing watches come first.
#[tauri::command]
#[specta::specta]
async fn stored_repos(state: State<'_, AppState>) -> Result<Vec<StoredRepoView>, String> {
    let repos = state.store.repos().await.map_err(|e| e.to_string())?;
    let pinned: std::collections::HashSet<String> = state
        .store
        .pinned_repo_ids()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();
    let discovered: std::collections::HashSet<String> = state
        .store
        .discovered_repo_ids()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();
    let mut out: Vec<StoredRepoView> = repos
        .into_iter()
        .map(|r| StoredRepoView {
            pinned: pinned.contains(&r.id),
            discovered: discovered.contains(&r.id),
            repo_id: r.id,
            full_name: r.full_name,
        })
        .collect();
    out.sort_by(|a, b| {
        let watched = |r: &StoredRepoView| r.pinned || r.discovered;
        watched(a)
            .cmp(&watched(b))
            .then_with(|| a.full_name.cmp(&b.full_name))
    });
    Ok(out)
}

/// Remove a repo and everything derived from it. Destructive and irreversible; the UI confirms
/// first. Refuses while a board pins the repo or a source still schedules it, and the error names
/// what holds it. A repo a board's person still owns is discovered again on the next pass.
#[tauri::command]
#[specta::specta]
async fn forget_repo(
    state: State<'_, AppState>,
    repo_id: String,
) -> Result<ForgottenRepoView, String> {
    let r = state
        .store
        .forget_repo(&repo_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(ForgottenRepoView {
        repo_id: r.repo_id,
        found: r.found,
        commits: r.commits as u32,
        pull_requests: r.pull_requests as u32,
        issues: r.issues as u32,
        code_files: r.code_files as u32,
        index_chunks: r.index_chunks as u32,
        total_rows: r.total_rows as u32,
    })
}

/// Storage accounting for the Debug view: boundary form of `core_store::StorageReport` plus the
/// on-disk assets the core cannot see. Byte figures are paired with measured/estimated.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct StorageReportView {
    /// "dbstat" (measured page counts) or "payload_estimate" (summed value lengths).
    pub method: String,
    /// True when the table breakdown came from real page counts.
    pub measured: bool,
    pub db_path: String,
    /// The database file's size on disk plus its WAL and shared-memory files. Null if in memory.
    pub file_bytes: Option<i64>,
    pub wal_bytes: Option<i64>,
    pub shm_bytes: Option<i64>,
    pub page_size: i64,
    pub page_count: i64,
    /// `page_count * page_size`: what SQLite claims inside the file. Measured either way.
    pub reserved_bytes: i64,
    /// Claimed pages holding nothing. Only a VACUUM returns them to the file system, so the file
    /// does not shrink after a delete.
    pub free_bytes: i64,
    pub attributed_bytes: i64,
    /// `reserved - attributed - free`. Near zero when measured; most of the file under the
    /// estimate, which cannot see indexes or page overhead.
    pub residual_bytes: i64,
    pub groups: Vec<StorageGroupView>,
    pub tables: Vec<StorageTableView>,
    /// Per repo. Always an estimate: SQLite attributes pages to tables, never to rows.
    pub repos: Vec<RepoStorageView>,
    /// Model files and directories outside the database.
    pub assets: Vec<AssetStorageView>,
    pub embed_dim: i64,
}

/// One storage group's share of the database.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct StorageGroupView {
    /// "search_index" | "chunk_text" | "activity" | "code" | "history" | "config" | "internal".
    pub group: String,
    pub bytes: i64,
    pub estimated: bool,
}

/// One table's share, with its indexes and (for vector / full-text tables) shadow tables.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct StorageTableView {
    pub name: String,
    pub group: String,
    pub bytes: i64,
    pub estimated: bool,
}

/// What one repo costs, estimated.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RepoStorageView {
    pub repo_id: String,
    pub full_name: String,
    pub commits: i64,
    pub pull_requests: i64,
    pub issues: i64,
    pub code_files: i64,
    pub index_chunks: i64,
    pub rows: i64,
    /// Summed byte length of the repo's stored text. Payload only, no indexes or page overhead.
    pub content_bytes: i64,
    /// `index_chunks * embed_dim * 4`: exact for the raw vectors, a floor on the index cost.
    pub vector_bytes: i64,
    /// `content_bytes + vector_bytes`. What to sort the list by.
    pub estimated_bytes: i64,
}

/// One on-disk asset outside the database: a model directory, or the app data dir itself.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AssetStorageView {
    pub name: String,
    /// "llm_downloaded" | "llm_bundled" | "embed_model" | "rerank_model" | "app_data".
    pub kind: String,
    pub path: String,
    pub exists: bool,
    pub bytes: i64,
    pub files: i64,
    /// True for an entry that contains other entries (the app data dir), so a UI does not add it
    /// into a total alongside them.
    pub includes_others: bool,
}

/// Total bytes and file count under `path`, walking nested directories. Symlinks are counted as
/// the link and not followed, so loops and double-counted symlinked model dirs cannot occur.
/// Unreadable entries are skipped; the file count shows how much was seen.
pub fn dir_bytes(path: &std::path::Path) -> (i64, i64) {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return (0, 0);
    };
    if !meta.is_dir() {
        return (meta.len() as i64, 1);
    }
    let (mut bytes, mut files) = (0i64, 0i64);
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            // `DirEntry::metadata` does not traverse symlinks, which keeps the walk finite.
            let Ok(m) = entry.metadata() else { continue };
            if m.is_dir() {
                stack.push(entry.path());
            } else {
                bytes += m.len() as i64;
                files += 1;
            }
        }
    }
    (bytes, files)
}

/// Describe one asset directory: path, whether it exists, and what it holds. A missing or
/// unreadable directory reports absent and zero. Bundled model dirs are absent in a dev build
/// without `fastembed`.
fn asset(
    name: &str,
    kind: &str,
    path: Option<&std::path::Path>,
    includes_others: bool,
) -> AssetStorageView {
    let (bytes, files) = path.map(dir_bytes).unwrap_or((0, 0));
    AssetStorageView {
        name: name.to_string(),
        kind: kind.to_string(),
        path: path
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "not configured".to_string()),
        exists: path.is_some_and(|p| p.exists()),
        bytes,
        files,
        includes_others,
    }
}

/// Measure what is on disk: the database, broken down, and the model assets beside it. Read-only.
/// Kept out of `debug_stats`: `dbstat` walks every page and the per-repo sums scan the payload
/// tables, while the Debug view re-runs `debug_stats` on every sync event.
#[tauri::command]
#[specta::specta]
async fn storage_report(state: State<'_, AppState>) -> Result<StorageReportView, String> {
    let r = state.store.storage().await.map_err(|e| e.to_string())?;
    let measured = r.method == core_store::StorageMethod::Dbstat;
    // The app data dir is the database's parent (see `main.rs`); it also holds downloaded models
    // and webview caches, so it explains disk use without attributing parts the app does not own.
    let app_data = std::path::Path::new(&state.db_path).parent();
    let assets = vec![
        asset(
            "Downloaded LLM models",
            "llm_downloaded",
            Some(&state.llm_download_dir),
            false,
        ),
        asset(
            "Bundled LLM model",
            "llm_bundled",
            state.llm_bundled_dir.as_deref(),
            false,
        ),
        asset(
            "Embedding model (ONNX)",
            "embed_model",
            state.embed_model_dir.as_deref(),
            false,
        ),
        asset(
            "Reranker model (ONNX)",
            "rerank_model",
            state.rerank_model_dir.as_deref(),
            false,
        ),
        asset("App data directory", "app_data", app_data, true),
    ];
    Ok(StorageReportView {
        method: match r.method {
            core_store::StorageMethod::Dbstat => "dbstat".into(),
            core_store::StorageMethod::PayloadEstimate => "payload_estimate".into(),
        },
        measured,
        db_path: state.db_path.clone(),
        file_bytes: r.file_bytes,
        wal_bytes: r.wal_bytes,
        shm_bytes: r.shm_bytes,
        page_size: r.page_size,
        page_count: r.page_count,
        reserved_bytes: r.reserved_bytes,
        free_bytes: r.free_bytes,
        attributed_bytes: r.attributed_bytes,
        residual_bytes: r.residual_bytes,
        groups: r
            .groups
            .iter()
            .map(|g| StorageGroupView {
                group: g.group.as_str().to_string(),
                bytes: g.bytes,
                estimated: g.estimated,
            })
            .collect(),
        tables: r
            .tables
            .iter()
            .map(|t| StorageTableView {
                name: t.name.clone(),
                group: t.group.as_str().to_string(),
                bytes: t.bytes,
                estimated: t.estimated,
            })
            .collect(),
        repos: r
            .repos
            .iter()
            .map(|p| RepoStorageView {
                repo_id: p.repo_id.clone(),
                full_name: p.full_name.clone(),
                commits: p.commits,
                pull_requests: p.pull_requests,
                issues: p.issues,
                code_files: p.code_files,
                index_chunks: p.index_chunks,
                rows: p.rows,
                content_bytes: p.content_bytes,
                vector_bytes: p.vector_bytes,
                estimated_bytes: p.estimated_bytes,
            })
            .collect(),
        assets,
        embed_dim: r.embed_dim,
    })
}

/// The live storage state: database cost, the user's ceiling, and whether the app is running,
/// warning or stopped. Cheap enough to poll (three pragmas, two `stat` calls, a walk of the model
/// dirs), unlike [`storage_report`], whose `dbstat` pass walks every page.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct StorageStateView {
    /// "normal" | "warning" | "full".
    pub pressure: String,
    /// True when syncing and indexing are stopped.
    pub stopped: bool,
    /// The plain-language sentence, built in the core so the banner and Settings card agree.
    pub message: String,
    /// The database file plus its write-ahead log: the figure the budget is enforced against.
    pub used_bytes: i64,
    /// The configured ceiling in bytes, or null when the user chose no limit.
    pub budget_bytes: Option<i64>,
    /// The same ceiling in megabytes, the unit the settings field edits.
    pub budget_mb: Option<i64>,
    /// Claimed pages holding nothing. A delete moves pages here; the file shrinks only on reclaim.
    pub free_bytes: i64,
    /// The model files on disk. Reported beside the budget, not inside it: models are fixed-size
    /// artifacts the user chose knowing their size.
    pub model_bytes: i64,
}

/// The live storage state. Read-only.
#[tauri::command]
#[specta::specta]
async fn storage_state(state: State<'_, AppState>) -> Result<StorageStateView, String> {
    let b = state
        .store
        .budget_state()
        .await
        .map_err(|e| e.to_string())?;
    // Only the model directories, not the app-data total: that also contains the database, so
    // adding it would count the same bytes twice.
    let model_bytes = [
        Some(state.llm_download_dir.as_path()),
        state.llm_bundled_dir.as_deref(),
        state.embed_model_dir.as_deref(),
        state.rerank_model_dir.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(|p| dir_bytes(p).0)
    .sum();
    Ok(StorageStateView {
        pressure: b.pressure.as_str().to_string(),
        stopped: !b.allows_sync(),
        message: b.message(),
        used_bytes: b.usage.used_bytes,
        budget_bytes: b.budget_bytes,
        budget_mb: b.budget_bytes.map(|bytes| bytes / core_store::MB),
        free_bytes: b.usage.free_bytes,
        model_bytes,
    })
}

/// What dropping one repo's search index removed. Every figure is rebuilt by the next sync.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DroppedIndexView {
    pub repo_id: String,
    pub found: bool,
    pub chunks: i64,
    pub code_files: i64,
    pub total_rows: i64,
}

/// Drop one repo's search index. Activity and metric snapshots stay; the next sync rebuilds it.
#[tauri::command]
#[specta::specta]
async fn drop_repo_index(
    state: State<'_, AppState>,
    repo_id: String,
) -> Result<DroppedIndexView, String> {
    let d = state
        .store
        .drop_repo_index(&repo_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(DroppedIndexView {
        repo_id: d.repo_id,
        found: d.found,
        chunks: d.chunks,
        code_files: d.code_files,
        total_rows: d.total_rows,
    })
}

/// What a reclaim did. No row count: it does not touch rows.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ReclaimReportView {
    pub used_before: i64,
    pub used_after: i64,
    pub freed_bytes: i64,
}

/// Give the free pages back to the file system: `VACUUM`, then truncate the log. Deletes nothing.
/// A delete does not shrink a SQLite file, so this is the only way under the ceiling after one.
/// Slow and exclusive: it needs room for a second copy while it holds the write lock.
#[tauri::command]
#[specta::specta]
async fn reclaim_free_space(state: State<'_, AppState>) -> Result<ReclaimReportView, String> {
    let r = state
        .store
        .reclaim_free_space()
        .await
        .map_err(|e| e.to_string())?;
    Ok(ReclaimReportView {
        used_before: r.used_before,
        used_after: r.used_after,
        freed_bytes: r.freed_bytes,
    })
}

/// One orphaned repo, sized as the Debug storage card sizes any repo.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct OrphanRepoView {
    pub repo_id: String,
    pub full_name: String,
    pub rows: i64,
    pub estimated_bytes: i64,
}

/// What no board can reach at all, pinned or discovered, before the user decides to reclaim it.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct OrphanRepoReportView {
    pub repos: Vec<OrphanRepoView>,
    pub count: u32,
    pub rows: i64,
    pub estimated_bytes: i64,
}

/// The orphan repo report: repos left behind by a deleted board's pin or discovery, which nothing
/// else cleans up. Read-only.
#[tauri::command]
#[specta::specta]
async fn orphan_repos(state: State<'_, AppState>) -> Result<OrphanRepoReportView, String> {
    let r = state
        .store
        .orphan_repos()
        .await
        .map_err(|e| e.to_string())?;
    Ok(OrphanRepoReportView {
        repos: r
            .repos
            .into_iter()
            .map(|p| OrphanRepoView {
                repo_id: p.repo_id,
                full_name: p.full_name,
                rows: p.rows,
                estimated_bytes: p.estimated_bytes,
            })
            .collect(),
        count: r.count as u32,
        rows: r.rows,
        estimated_bytes: r.estimated_bytes,
    })
}

/// What reclaiming every orphaned repo removed, the bulk form of `ForgottenRepoView`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct OrphanReclaimView {
    pub repo_ids: Vec<String>,
    pub count: u32,
    pub total_rows: u32,
}

/// Remove every repo no board references, and everything derived from each: `forget_repo`'s
/// removal for the whole orphan set in one transaction. Destructive and irreversible. Never
/// refuses: each repo is already unpinned and undiscovered by construction.
#[tauri::command]
#[specta::specta]
async fn reclaim_orphan_repos(state: State<'_, AppState>) -> Result<OrphanReclaimView, String> {
    let r = state
        .store
        .reclaim_orphan_repos()
        .await
        .map_err(|e| e.to_string())?;
    Ok(OrphanReclaimView {
        repo_ids: r.repo_ids,
        count: r.count as u32,
        total_rows: r.total_rows as u32,
    })
}

/// Snapshot the local-LLM narrator status: feature built in, model configured and present,
/// loaded this session, and the last engine error.
fn llm_status() -> LlmStatusView {
    let dir = std::env::var("ORGONZOLA_LLM_DIR")
        .ok()
        .filter(|s| !s.is_empty());
    let file = std::env::var("ORGONZOLA_LLM_GGUF")
        .ok()
        .filter(|s| !s.is_empty());
    let configured = dir.is_some() && file.is_some();
    let model_path = match (&dir, &file) {
        (Some(d), Some(f)) => Some(format!("{d}/{f}")),
        _ => None,
    };
    let model_present = model_path
        .as_deref()
        .map(|p| std::path::Path::new(p).is_file())
        .unwrap_or(false);
    LlmStatusView {
        feature_built: core_llm::feature_enabled(),
        configured,
        model_path,
        model_present,
        loaded: core_llm::is_loaded(),
        last_error: core_llm::last_error(),
    }
}

/// The repos a board's Search tab may answer from: its effective repo set (pinned if any,
/// otherwise discovered), the same set the board's other tabs and the assistant use.
/// An unknown board id is an error, not an unscoped search over the whole store. A board with no
/// repos yields an empty set, and `RepoScope::Repos(&[])` matches nothing.
async fn board_search_scope(store: &Store, board_id: &str) -> Result<Vec<String>, String> {
    if store
        .board(board_id)
        .await
        .map_err(|e| e.to_string())?
        .is_none()
    {
        return Err(format!("no board {board_id}"));
    }
    store
        .board_effective_repo_ids(board_id)
        .await
        .map_err(|e| e.to_string())
}

/// Semantic search over indexed activity text: hybrid retrieval (vector + keyword) then reranking,
/// excluding code chunks, within the board's repos. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn semantic_search(
    state: State<'_, AppState>,
    board_id: String,
    query: String,
    k: i64,
) -> Result<Vec<SearchHit>, String> {
    let repos = board_search_scope(&state.store, &board_id).await?;
    search_and_rerank(
        &state,
        &query,
        k,
        core_store::KindFilter::Not("code"),
        core_store::RepoScope::Repos(&repos),
    )
    .await
}

/// Semantic search over indexed source code: hybrid retrieval then reranking, restricted to `code`
/// chunks (file path + snippet) in the board's repos. The code index is filled by
/// `SyncEngine::sync_code`. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn semantic_code_search(
    state: State<'_, AppState>,
    board_id: String,
    query: String,
    k: i64,
) -> Result<Vec<SearchHit>, String> {
    let repos = board_search_scope(&state.store, &board_id).await?;
    search_and_rerank(
        &state,
        &query,
        k,
        core_store::KindFilter::Is("code"),
        core_store::RepoScope::Repos(&repos),
    )
    .await
}

/// A pull-request reference on a ritual agenda. Boundary form of `core_ritual::PrRef`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PrRefView {
    pub id: String,
    pub number: i64,
    pub title: String,
    pub url: Option<String>,
}

impl From<core_ritual::PrRef> for PrRefView {
    fn from(p: core_ritual::PrRef) -> Self {
        Self {
            id: p.id,
            number: p.number,
            title: p.title,
            url: p.url,
        }
    }
}

/// Application settings for the UI. Boundary form of `core_store::Settings`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SettingsView {
    pub sync_period_secs: i64,
    pub stale_pr_days: i64,
    /// Slack-compatible incoming-webhook URL for the digest push, or null if unset.
    pub digest_webhook_url: Option<String>,
    /// The scheduled-digest cadence in hours, or null if off.
    pub digest_schedule_hours: Option<i64>,
    /// Whether local-LLM narration is enabled at runtime.
    pub llm_enabled: bool,
    /// The selected GGUF model filename, or null to use the env default.
    pub llm_model: Option<String>,
    /// Disk the database may use, in megabytes, or null for no limit.
    pub storage_budget_mb: Option<i64>,
}

impl From<Settings> for SettingsView {
    fn from(s: Settings) -> Self {
        Self {
            sync_period_secs: s.sync_period_secs,
            stale_pr_days: s.stale_pr_days,
            digest_webhook_url: s.digest_webhook_url,
            digest_schedule_hours: s.digest_schedule_hours,
            llm_enabled: s.llm_enabled,
            llm_model: s.llm_model,
            storage_budget_mb: s.storage_budget_mb,
        }
    }
}

/// The current application settings. Read-only; no token.
#[tauri::command]
#[specta::specta]
async fn get_settings(state: State<'_, AppState>) -> Result<SettingsView, String> {
    let s = state.store.settings().await.map_err(|e| e.to_string())?;
    Ok(s.into())
}

/// Update the application settings. `sync_period_secs` is the scheduler poll cadence (>= 1s,
/// applied on the next pass); `stale_pr_days` is the open-PR staleness threshold (>= 1).
/// Returns the new values.
#[tauri::command]
#[specta::specta]
async fn set_settings(
    state: State<'_, AppState>,
    sync_period_secs: i64,
    stale_pr_days: i64,
    digest_webhook_url: Option<String>,
    digest_schedule_hours: Option<i64>,
    storage_budget_mb: Option<i64>,
) -> Result<SettingsView, String> {
    if sync_period_secs < 1 {
        return Err("sync period must be at least 1 second".into());
    }
    if stale_pr_days < 1 {
        return Err("stale-PR threshold must be at least 1 day".into());
    }
    // A blank URL means "unset", so it is not stored (egress would use it).
    let digest_webhook_url = digest_webhook_url.filter(|u| !u.trim().is_empty());
    if let Some(url) = &digest_webhook_url {
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err("the digest webhook must be an http(s) URL".into());
        }
    }
    // A schedule of 0 or less means off; only a positive cadence is stored.
    let digest_schedule_hours = digest_schedule_hours.filter(|h| *h > 0);
    // Preserve the LLM settings, which have their own command.
    let current = state.store.settings().await.map_err(|e| e.to_string())?;
    let s = Settings {
        sync_period_secs,
        stale_pr_days,
        digest_webhook_url,
        digest_schedule_hours,
        llm_enabled: current.llm_enabled,
        llm_model: current.llm_model,
        // A budget of zero or less means "no limit"; a stored zero would read as "stop at once".
        // The UI sends null for no limit; this covers a caller that sends 0.
        storage_budget_mb: storage_budget_mb.filter(|mb| *mb > 0),
    };
    state
        .store
        .update_settings(&s)
        .await
        .map_err(|e| e.to_string())?;
    Ok(s.into())
}

// ---- local-LLM settings + model management ----

/// One model in the curated local-LLM catalog: a small instruct GGUF that runs on CPU.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct LlmModelView {
    pub label: String,
    pub repo: String,
    pub file: String,
    pub params: String,
    pub size: String,
    /// Whether the GGUF is present in the models dir.
    pub downloaded: bool,
}

/// The local-LLM settings and catalog for the Settings tab.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct LlmCatalogView {
    /// Whether the `llm` feature is compiled into this binary.
    pub feature_built: bool,
    /// The runtime on/off switch.
    pub enabled: bool,
    /// The selected GGUF filename, or null (env default).
    pub selected: Option<String>,
    /// Where models are stored and downloaded to.
    pub dir: String,
    pub models: Vec<LlmModelView>,
}

/// The curated small-model catalog: CPU-friendly instruct GGUFs with verified HF paths.
const LLM_CATALOG: [(&str, &str, &str, &str, &str); 3] = [
    (
        "Qwen3 0.6B",
        "unsloth/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-Q4_K_M.gguf",
        "0.6B",
        "~0.4 GB",
    ),
    (
        "Qwen3 1.7B",
        "unsloth/Qwen3-1.7B-GGUF",
        "Qwen3-1.7B-Q4_K_M.gguf",
        "1.7B",
        "~1.1 GB",
    ),
    (
        "Gemma 3 1B",
        "unsloth/gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q4_K_M.gguf",
        "1B",
        "~0.8 GB",
    ),
];

/// The local-LLM settings and model catalog. Read-only; no token.
#[tauri::command]
#[specta::specta]
async fn llm_catalog(state: State<'_, AppState>) -> Result<LlmCatalogView, String> {
    let s = state.store.settings().await.map_err(|e| e.to_string())?;
    let models = LLM_CATALOG
        .iter()
        .map(|(label, repo, file, params, size)| LlmModelView {
            label: label.to_string(),
            repo: repo.to_string(),
            file: file.to_string(),
            params: params.to_string(),
            size: size.to_string(),
            downloaded: state.llm_dir_for(file).is_some(),
        })
        .collect();
    Ok(LlmCatalogView {
        feature_built: core_llm::feature_enabled(),
        enabled: s.llm_enabled,
        selected: s.llm_model,
        dir: state.llm_download_dir.to_string_lossy().into_owned(),
        models,
    })
}

/// Set the local-LLM runtime switch and selected model. Returns the updated settings.
#[tauri::command]
#[specta::specta]
async fn set_llm_settings(
    state: State<'_, AppState>,
    enabled: bool,
    model: Option<String>,
) -> Result<SettingsView, String> {
    let mut s = state.store.settings().await.map_err(|e| e.to_string())?;
    s.llm_enabled = enabled;
    s.llm_model = model.filter(|m| !m.trim().is_empty());
    state
        .store
        .update_settings(&s)
        .await
        .map_err(|e| e.to_string())?;
    Ok(s.into())
}

/// Progress of a local-LLM model download. `done`/`error` terminate it.
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
pub struct LlmDownloadEvent {
    pub file: String,
    pub received: u64,
    pub total: u64,
    pub done: bool,
    pub error: Option<String>,
}

/// Download a GGUF model from Hugging Face into the writable models dir, streamed to disk and
/// emitting `LlmDownloadEvent` progress. Writes a `.part` file, then renames on success. Egress
/// happens only when the user clicks Download.
#[tauri::command]
#[specta::specta]
async fn download_llm_model(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    repo: String,
    file: String,
) -> Result<(), String> {
    use futures_util::StreamExt;
    use std::io::Write;

    let dir = state.llm_download_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let dest = dir.join(&file);
    let part = dest.with_extension("gguf.part");
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");

    let emit = |received: u64, total: u64, done: bool, error: Option<String>| {
        let _ = LlmDownloadEvent {
            file: file.clone(),
            received,
            total,
            done,
            error,
        }
        .emit(&app);
    };

    let resp = match reqwest::get(&url).await.and_then(|r| r.error_for_status()) {
        Ok(r) => r,
        Err(e) => {
            emit(0, 0, true, Some(e.to_string()));
            return Err(e.to_string());
        }
    };
    let total = resp.content_length().unwrap_or(0);
    let mut out = match std::fs::File::create(&part) {
        Ok(f) => f,
        Err(e) => {
            emit(0, total, true, Some(e.to_string()));
            return Err(e.to_string());
        }
    };
    let mut received = 0u64;
    let mut next_tick = 0u64;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                emit(received, total, true, Some(e.to_string()));
                return Err(e.to_string());
            }
        };
        if let Err(e) = out.write_all(&chunk) {
            emit(received, total, true, Some(e.to_string()));
            return Err(e.to_string());
        }
        received += chunk.len() as u64;
        // Throttle progress events to about every 4 MB.
        if received >= next_tick {
            emit(received, total, false, None);
            next_tick = received + 4 * 1024 * 1024;
        }
    }
    drop(out);
    if let Err(e) = std::fs::rename(&part, &dest) {
        emit(received, total, true, Some(e.to_string()));
        return Err(e.to_string());
    }
    emit(received, total, true, None);
    Ok(())
}

/// Open a web URL in the user's default browser. The webview does not follow `target="_blank"`
/// itself, so the UI hands outbound forge links here. Only http(s) is allowed.
#[tauri::command]
#[specta::specta]
async fn open_url(url: String) -> Result<(), String> {
    let url = url.trim().to_string();
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("refusing to open a non-http(s) URL".into());
    }
    tokio::task::spawn_blocking(move || open::that(&url))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

/// Sync every configured source now, independent of the scheduler's cadence. Emits a
/// `SyncProgressEvent` (with `done`/`total`) as each source completes, then returns the finished
/// list.
#[tauri::command]
#[specta::specta]
async fn sync_now(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<SyncProgressEvent>, String> {
    let engine = state.engine().ok_or(NO_CONNECTION)?;
    // Emit every progress update live; return only the per-repo completion events.
    let mut finished = Vec::new();
    engine
        .sync_all_sources(|p| {
            let event = SyncProgressEvent::from(p);
            let _ = event.emit(&app);
            if event.finished {
                finished.push(event);
            }
        })
        .await
        // Plain-language, actionable message instead of a raw status code.
        .map_err(|e| e.user_message())?;
    // Record a daily metric snapshot for every synced repo.
    record_repo_snapshots(&state.store).await;
    // Read-only Jira sync for boards with linked projects: issues, sprints, PR linking.
    jira::sync_jira(&state.store, state.credentials.as_ref()).await;
    Ok(finished)
}

/// Record a daily metric snapshot for every synced repo, dated with the host's clock.
/// Best-effort: a per-repo failure does not stop the others. Public for the scheduler task.
pub async fn record_repo_snapshots(store: &Store) {
    let now = now_rfc3339();
    let Ok(repos) = store.repos().await else {
        return;
    };
    for repo in repos {
        let _ = core_summary::record_snapshot(store, "repo", &repo.id, &now).await;
    }
}

/// Record a daily metric snapshot for one repo (the scheduler calls this per polled source).
pub async fn record_repo_snapshot(store: &Store, repo_id: &str) {
    let _ = core_summary::record_snapshot(store, "repo", repo_id, &now_rfc3339()).await;
}

/// Record today's board-scoped flow and bug metrics for every board. Idempotent per day (the
/// snapshot store overwrites today's key). Best-effort. `lead_time` is recorded only when there
/// is a completed item, never a fake zero.
pub async fn record_board_snapshots(store: &Store) {
    let now = now_rfc3339();
    let day = &now[..now.len().min(10)]; // YYYY-MM-DD
    let Ok(boards) = store.boards().await else {
        return;
    };
    for board in boards {
        let Ok(repo_ids) = store.board_effective_repo_ids(&board.id).await else {
            continue;
        };
        if let Ok(flow) =
            core_summary::board_flow_metrics(store, &repo_ids, &days_ago_rfc3339(30), &now, &now)
                .await
        {
            let _ = store
                .record_metric("board", &board.id, day, "flow_wip", flow.in_progress as i64)
                .await;
            if let Some(lead) = flow.median_lead_time_secs {
                let _ = store
                    .record_metric("board", &board.id, day, "flow_lead_secs", lead)
                    .await;
            }
        }
        if let Ok(bugs) =
            core_summary::board_bug_flow(store, &repo_ids, &days_ago_rfc3339(30), &now).await
        {
            let _ = store
                .record_metric(
                    "board",
                    &board.id,
                    day,
                    "bug_net",
                    bugs.opened as i64 - bugs.closed as i64,
                )
                .await;
        }
    }
}

// ---- boards ----

/// A board for the UI. Boundary form of `core_store::Board`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardView {
    pub id: String,
    pub name: String,
    /// What the board watches: "team" | "repo" | "org". Drives scope inputs, visible tabs and
    /// indexing profile. Fixed at creation.
    pub kind: String,
    /// GitHub logins on the board (the people watched).
    pub people: Vec<String>,
    /// `owner/name` of each pinned repo. Empty = the board covers everything its people touch.
    pub repos: Vec<String>,
    /// Optional org restriction: only repos in this org are discovered. `None` = no limit.
    pub org: Option<String>,
    /// The forge this board syncs from. `None` = unassigned; the board cannot be synced.
    pub forge_id: Option<String>,
    /// Opt-in dependency vulnerability scanning. Off by default.
    pub scan_dependencies: bool,
    /// Whether a team board's people-derived discovery includes archived repositories. Off by
    /// default; direct repository pins bypass this.
    pub include_archived: bool,
}

impl From<Board> for BoardView {
    fn from(b: Board) -> Self {
        Self {
            id: b.id,
            name: b.name,
            kind: b.kind.as_str().to_string(),
            people: b.people,
            // Present repos as owner/name, not the internal "repo:<forge>/owner/name" id.
            repos: b.repos.iter().map(|r| full_name_of_repo_id(r)).collect(),
            org: b.org,
            forge_id: b.forge_id,
            scan_dependencies: b.scan_dependencies,
            include_archived: b.include_archived,
        }
    }
}

/// The `owner/name` part of a namespaced repo id (`repo:<forge>/<owner>/<name>`), for display.
/// Falls back to the raw string.
fn full_name_of_repo_id(id: &str) -> String {
    id.strip_prefix("repo:")
        .and_then(|s| s.split_once('/'))
        .map(|(_forge, full)| full.to_string())
        .unwrap_or_else(|| id.to_string())
}

/// Slugify a board name into a stable id, e.g. "Platform Team" -> "board:platform-team".
fn board_id_from_name(name: &str) -> String {
    let slug: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let slug = slug.trim_matches('-').to_string();
    format!("board:{}", if slug.is_empty() { "board" } else { &slug })
}

/// Fetch a board or return a user-facing error if it does not exist.
async fn require_board(store: &Store, id: &str) -> Result<BoardView, String> {
    store
        .board(id)
        .await
        .map_err(|e| e.to_string())?
        .map(BoardView::from)
        .ok_or_else(|| format!("no board {id}"))
}

/// Create a board of the given kind ("team" | "repo" | "org") on the given forge. The forge is
/// required and fixed for the board's life. For a repo/org board, `target` is its one repo
/// (`owner/name`) or org login: it is set immediately, names the board (`name` is ignored) and
/// cannot change later. A team board takes `name` and starts with empty, editable scope.
/// Idempotent by the name's slug id. Returns the board.
#[tauri::command]
#[specta::specta]
async fn create_board(
    state: State<'_, AppState>,
    name: String,
    kind: String,
    forge_id: String,
    target: Option<String>,
) -> Result<BoardView, String> {
    let kind = match kind.as_str() {
        "team" => BoardKind::Team,
        "repo" => BoardKind::Repo,
        "org" => BoardKind::Org,
        other => return Err(format!("unknown board kind {other:?}")),
    };
    let forge_id = forge_id.trim();
    if forge_id.is_empty() {
        return Err("pick a forge for this board".into());
    }
    if state
        .store
        .forge(forge_id)
        .await
        .map_err(|e| e.to_string())?
        .is_none()
    {
        return Err(format!("no forge {forge_id}"));
    }
    let target = target
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    // A repo/org board is named after its target; a team board takes the typed name.
    let name = match kind {
        BoardKind::Team => name.trim().to_string(),
        BoardKind::Repo => target.clone().ok_or("pick a repository for this board")?,
        BoardKind::Org => target.clone().ok_or("pick an org for this board")?,
    };
    if name.is_empty() {
        return Err("board name cannot be empty".into());
    }
    let id = board_id_from_name(&name);
    state
        .store
        .create_board(&id, &name, kind, &now_rfc3339())
        .await
        .map_err(|e| e.to_string())?;
    state
        .store
        .set_board_forge(&id, Some(forge_id))
        .await
        .map_err(|e| e.to_string())?;
    match kind {
        BoardKind::Repo => pin_board_repo(&state.store, &id, forge_id, &name).await?,
        BoardKind::Org => state
            .store
            .set_board_org(&id, Some(&name))
            .await
            .map_err(|e| e.to_string())?,
        BoardKind::Team => {}
    }
    require_board(&state.store, &id).await
}

/// Every board, each with its people and pinned repos.
#[tauri::command]
#[specta::specta]
async fn list_boards(state: State<'_, AppState>) -> Result<Vec<BoardView>, String> {
    let boards = state.store.boards().await.map_err(|e| e.to_string())?;
    Ok(boards.into_iter().map(BoardView::from).collect())
}

/// Delete a board (its members go with it).
#[tauri::command]
#[specta::specta]
async fn delete_board(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state
        .store
        .delete_board(&id)
        .await
        .map_err(|e| e.to_string())
}

/// Add a person (GitHub login) to a board. Returns the updated board.
#[tauri::command]
#[specta::specta]
async fn add_board_person(
    state: State<'_, AppState>,
    id: String,
    login: String,
) -> Result<BoardView, String> {
    let login = login.trim().trim_start_matches('@');
    if login.is_empty() {
        return Err("login cannot be empty".into());
    }
    state
        .store
        .add_board_person(&id, login)
        .await
        .map_err(|e| e.to_string())?;
    require_board(&state.store, &id).await
}

/// Remove a person from a board. Returns the updated board.
#[tauri::command]
#[specta::specta]
async fn remove_board_person(
    state: State<'_, AppState>,
    id: String,
    login: String,
) -> Result<BoardView, String> {
    state
        .store
        .remove_board_person(&id, login.trim())
        .await
        .map_err(|e| e.to_string())?;
    require_board(&state.store, &id).await
}

/// Pin a repo (`owner/name`) to a board and register it as a synced source. Returns the board.
#[tauri::command]
#[specta::specta]
async fn add_board_repo(
    state: State<'_, AppState>,
    id: String,
    full_name: String,
) -> Result<BoardView, String> {
    let forge_id = board_forge_id(&state.store, &id).await?;
    pin_board_repo(&state.store, &id, &forge_id, full_name.trim()).await?;
    require_board(&state.store, &id).await
}

/// Pin `full_name` (`owner/name`) to a board: register it as a synced source if new, then add it
/// to the board's repos. Namespaced by the board's forge so it routes to the right backend and
/// does not collide with the same owner/name on another forge. Shared with the contributor import.
async fn pin_board_repo(
    store: &Store,
    board_id: &str,
    forge_id: &str,
    full_name: &str,
) -> Result<(), String> {
    if full_name.split_once('/').is_none() {
        return Err("repo must be in owner/name form".into());
    }
    let repo_id = namespaced_repo_id(forge_id, full_name);
    // Register it as a synced source if not already there.
    let sources = store.sources().await.map_err(|e| e.to_string())?;
    if !sources.iter().any(|s| s.id == repo_id) {
        store
            .upsert_source(&SourceRow {
                id: repo_id.clone(),
                kind: "repo".into(),
                name: full_name.to_string(),
                ownership: "owned".into(),
                filters: Vec::new(),
                stale_pr_days: 7,
                forge_id: Some(forge_id.to_string()),
            })
            .await
            .map_err(|e| e.to_string())?;
    }
    store
        .add_board_repo(board_id, &repo_id)
        .await
        .map_err(|e| e.to_string())
}

/// Search a forge for users matching `query`, for the source pickers. Forge-scoped so it works
/// before any board exists. Read-only.
#[tauri::command]
#[specta::specta]
async fn search_forge_users(
    state: State<'_, AppState>,
    forge_id: String,
    query: String,
) -> Result<Vec<String>, String> {
    let Some((engine, q)) = search_context(&state, &forge_id, &query)? else {
        return Ok(Vec::new());
    };
    engine
        .search_users(&forge_id, &q)
        .await
        .map_err(|e| e.to_string())
}

/// Search a forge for orgs matching `query`. See [`search_forge_users`].
#[tauri::command]
#[specta::specta]
async fn search_forge_orgs(
    state: State<'_, AppState>,
    forge_id: String,
    query: String,
) -> Result<Vec<String>, String> {
    let Some((engine, q)) = search_context(&state, &forge_id, &query)? else {
        return Ok(Vec::new());
    };
    engine
        .search_orgs(&forge_id, &q)
        .await
        .map_err(|e| e.to_string())
}

/// Search a forge for repos (`owner/name`) matching `query`. See [`search_forge_users`].
#[tauri::command]
#[specta::specta]
async fn search_forge_repos(
    state: State<'_, AppState>,
    forge_id: String,
    query: String,
) -> Result<Vec<String>, String> {
    let Some((engine, q)) = search_context(&state, &forge_id, &query)? else {
        return Ok(Vec::new());
    };
    engine
        .search_repos(&forge_id, &q)
        .await
        .map_err(|e| e.to_string())
}

/// Resolve what the three source-picker searches share: the engine and the trimmed query.
/// `Ok(None)` means only a blank query. Every other reason the search cannot run is an `Err`
/// naming it, so the caller can tell an empty result from a search that never ran.
fn search_context(
    state: &AppState,
    forge_id: &str,
    query: &str,
) -> Result<Option<(Arc<DesktopSyncEngine>, String)>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(None);
    }
    if forge_id.trim().is_empty() {
        return Err("this board has no connection yet - pick one in the board's Settings".into());
    }
    let engine = state.engine().ok_or(NO_CONNECTION)?;
    if !engine
        .forge_capabilities(forge_id)
        .map_err(|e| e.to_string())?
        .search
    {
        return Err("this connection has no search API - type the exact name".into());
    }
    Ok(Some((engine, query.to_string())))
}

/// The forge a board syncs from, or a user-facing error if none is assigned.
async fn board_forge_id(store: &Store, board_id: &str) -> Result<String, String> {
    store
        .board(board_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no board {board_id}"))?
        .forge_id
        .ok_or_else(|| "assign a forge to this board first".to_string())
}

/// Unpin a repo from a board. If no other board pins it, it stops being a synced source.
/// Returns the updated board.
#[tauri::command]
#[specta::specta]
async fn remove_board_repo(
    state: State<'_, AppState>,
    id: String,
    full_name: String,
) -> Result<BoardView, String> {
    let forge_id = board_forge_id(&state.store, &id).await?;
    let repo_id = namespaced_repo_id(&forge_id, full_name.trim());
    state
        .store
        .remove_board_repo(&id, &repo_id)
        .await
        .map_err(|e| e.to_string())?;
    require_board(&state.store, &id).await
}

/// Set or clear a board's org restriction (an empty string clears it). When set, only repos in
/// that org are discovered from the board's people. Returns the updated board.
#[tauri::command]
#[specta::specta]
async fn set_board_org(
    state: State<'_, AppState>,
    id: String,
    org: Option<String>,
) -> Result<BoardView, String> {
    let org = org.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    state
        .store
        .set_board_org(&id, org.as_deref())
        .await
        .map_err(|e| e.to_string())?;
    require_board(&state.store, &id).await
}

/// Toggle whether a team board's people-derived discovery includes archived repositories.
/// Explicitly pinned repositories are always in scope. Returns the updated board.
#[tauri::command]
#[specta::specta]
async fn set_board_include_archived(
    state: State<'_, AppState>,
    id: String,
    include_archived: bool,
) -> Result<BoardView, String> {
    state
        .store
        .set_board_include_archived(&id, include_archived)
        .await
        .map_err(|e| e.to_string())?;
    require_board(&state.store, &id).await
}

/// Toggle whether a repo's source code is indexed for code search. Off skips the code fetch and
/// embed and drops the repo's existing code index. Returns the new value.
#[tauri::command]
#[specta::specta]
async fn set_repo_index_code(
    state: State<'_, AppState>,
    repo_id: String,
    enabled: bool,
) -> Result<bool, String> {
    state
        .store
        .set_repo_index_code(&repo_id, enabled)
        .await
        .map_err(|e| e.to_string())?;
    if !enabled {
        state
            .store
            .clear_repo_code_index(&repo_id)
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(enabled)
}

/// The board's effective repos as `Repo` rows (pinned, or the people's discovered repos).
async fn board_repos_resolved(store: &Store, board_id: &str) -> Result<Vec<Repo>, String> {
    let ids = store
        .board_effective_repo_ids(board_id)
        .await
        .map_err(|e| e.to_string())?;
    let by_id: std::collections::HashMap<String, Repo> = store
        .repos()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|r| (r.id.clone(), r))
        .collect();
    Ok(ids
        .into_iter()
        .filter_map(|id| by_id.get(&id).cloned())
        .collect())
}

/// A board-level attention item. Boundary form of `core_summary::BoardAttentionItem`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardAttentionView {
    pub repo_id: String,
    pub kind: String,
    /// The item's typed envelope, the same shape as the per-repo `AttentionView`.
    pub entities: Vec<core_model::LinkedEntity>,
    pub summary: String,
    pub by_team: bool,
    /// For `aging_wip` items: band of the board's in-progress-time distribution.
    pub wip_percentile: Option<String>,
    /// What `wip_percentile` is measured over: span, window and sample size.
    pub wip_percentile_basis: Option<String>,
}

impl From<core_summary::BoardAttentionItem> for BoardAttentionView {
    fn from(i: core_summary::BoardAttentionItem) -> Self {
        Self {
            repo_id: i.repo_id,
            kind: i.kind,
            entities: i.entities,
            summary: i.summary,
            by_team: i.by_team,
            wip_percentile: i.wip_percentile,
            wip_percentile_basis: i.wip_percentile_basis,
        }
    }
}

/// Attention rolled up across a board: the union of its repos' items, with the board's people
/// flagged. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn board_attention(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<BoardAttentionView>, String> {
    let Some(board) = state.store.board(&id).await.map_err(|e| e.to_string())? else {
        return Err(format!("no board {id}"));
    };
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let disabled = state
        .store
        .board_disabled_signals(&id)
        .await
        .map_err(|e| e.to_string())?;
    let items =
        core_summary::board_attention(&state.store, &repo_ids, &board.people, &now_rfc3339())
            .await
            .map_err(|e| e.to_string())?;
    Ok(items
        .into_iter()
        .filter(|i| !disabled.contains(&i.kind))
        .map(BoardAttentionView::from)
        .collect())
}

/// The attention narrator's system prompt. The UI already shows per-kind counts as chips, so the
/// model should synthesize what to focus on first, not restate totals. Facts only, no invented
/// items. It says explicitly not to write a list, because the local model mirrors list-shaped
/// input (see `attention_focus_input`).
const ATTENTION_SYSTEM: &str = "You are an assistant to an engineering manager. From the attention items \
below, write two or three short, plain sentences saying what to focus on first and why, pointing to the most \
pressing specific item(s). Do NOT just list the counts - the manager can already see those. Do NOT write a \
list or one line per item - write flowing prose sentences. Use ONLY the facts given; do not invent items, \
counts, names, or numbers. Be concrete and terse.";

/// One streamed step of the AI brief, tagged with the board and the `run_id` of the
/// `start_board_brief` call, so the UI accepts only its own stream: one board can have two
/// overlapping generations in flight (StrictMode double mount) and their tokens would interleave.
/// `phase` is "loading"/"prefilling"/"thinking" (before the first token), "delta" (a token in
/// `delta`), "done", or "disabled" (no model or `llm` feature off).
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
pub struct BriefEvent {
    pub run_id: String,
    pub board_id: String,
    pub phase: String,
    pub delta: String,
}

impl BriefEvent {
    fn phase(run_id: &str, board_id: &str, phase: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            board_id: board_id.to_string(),
            phase: phase.to_string(),
            delta: String::new(),
        }
    }
}

/// Stream the board's AI brief: rephrase the deterministic attention brief with the local model,
/// emitting `BriefEvent`s (thinking -> delta* -> done). When the `llm` feature is off or no model
/// is configured, emits a single `disabled`. Local-first; no key, no network. Returns once
/// generation ends.
#[tauri::command]
#[specta::specta]
async fn start_board_brief(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    id: String,
    run_id: String,
) -> Result<(), String> {
    if !cfg!(feature = "llm") {
        let _ = BriefEvent::phase(&run_id, &id, "disabled").emit(&app);
        return Ok(());
    }
    let settings = state.store.settings().await.map_err(|e| e.to_string())?;
    // The selected model (Settings), else the env default that `just run` sets.
    let file = settings
        .llm_model
        .clone()
        .or_else(|| std::env::var("ORGONZOLA_LLM_GGUF").ok())
        .filter(|f| !f.is_empty());
    // Locate it across the download dir and the bundled dir; `None` means not present.
    let dir = file.as_deref().and_then(|f| state.llm_dir_for(f));
    if !settings.llm_enabled || dir.is_none() {
        let _ = BriefEvent::phase(&run_id, &id, "disabled").emit(&app);
        return Ok(());
    }
    let dir = dir.unwrap_or_default();
    let file = file.unwrap_or_default();
    let Some(board) = state.store.board(&id).await.map_err(|e| e.to_string())? else {
        return Err(format!("no board {id}"));
    };
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let disabled = state
        .store
        .board_disabled_signals(&id)
        .await
        .map_err(|e| e.to_string())?;
    let now = now_rfc3339();
    let items = core_summary::board_attention(&state.store, &repo_ids, &board.people, &now)
        .await
        .map_err(|e| e.to_string())?;
    let items: Vec<_> = items
        .into_iter()
        .filter(|i| !disabled.contains(&i.kind))
        .collect();
    // Feed the model the pressing items with severity and age, not just the counts, so it can rank
    // instead of calling everything urgent.
    let brief = core_summary::attention_focus_input(&items, 6, &now);
    let messages = vec![
        core_llm::ChatMessage::system(ATTENTION_SYSTEM),
        core_llm::ChatMessage::user(brief),
    ];

    // Phases for the Debug panel: loading (first-run GGUF load) -> prefilling (prompt processing)
    // -> delta* (token generation) -> done.
    if !core_llm::is_loaded() {
        let _ = BriefEvent::phase(&run_id, &id, "loading").emit(&app);
    }
    if core_llm::ensure_loaded(&dir, &file).await.is_err() {
        let _ = BriefEvent::phase(&run_id, &id, "disabled").emit(&app);
        return Ok(());
    }
    let _ = BriefEvent::phase(&run_id, &id, "prefilling").emit(&app);

    let token_app = app.clone();
    let token_board = id.clone();
    let token_run = run_id.clone();
    let result = core_llm::stream_complete(&dir, &file, messages, |tok| {
        let _ = BriefEvent {
            run_id: token_run.clone(),
            board_id: token_board.clone(),
            phase: "delta".to_string(),
            delta: tok.to_string(),
        }
        .emit(&token_app);
        true // the narrator always runs to completion
    })
    .await;

    let _ = match result {
        Ok(()) => BriefEvent::phase(&run_id, &id, "done").emit(&app),
        // Model load or generation failed: hide the card rather than show a partial line.
        Err(_) => BriefEvent::phase(&run_id, &id, "disabled").emit(&app),
    };
    Ok(())
}

// ---- AI assistant ----

/// One streamed step of an assistant chat turn, tagged with the `run_id` of the
/// `start_assistant_chat` call so overlapping turns do not interleave. `phase` is "thinking"
/// (running tools or generating), "delta" (a piece of the answer in `delta`), "done", "disabled"
/// (feature off or no model), or "error" (the agent failed mid-turn; `delta` carries the message).
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
pub struct AssistantChatEvent {
    pub run_id: String,
    pub phase: String,
    pub delta: String,
}

impl AssistantChatEvent {
    fn phase(run_id: &str, phase: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            phase: phase.to_string(),
            delta: String::new(),
        }
    }
}

/// One prior turn of the conversation, sent from the UI for context. `role` is "user" or
/// "assistant".
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ChatTurnView {
    pub role: String,
    pub content: String,
}

/// Answer a question with the conversational assistant: a Rig agent over the local model that
/// calls read-only KB tools scoped to `board_id` and grounds its reply via RAG. Streams
/// `AssistantChatEvent` (thinking -> delta -> done), or a single `disabled` when the assistant is
/// not built in or no model is ready. Local-first; no key, no network.
#[tauri::command]
#[specta::specta]
async fn start_assistant_chat(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    run_id: String,
    board_id: String,
    history: Vec<ChatTurnView>,
    question: String,
    tab: String,
) -> Result<(), String> {
    let settings = state.store.settings().await.map_err(|e| e.to_string())?;
    let file = settings
        .llm_model
        .clone()
        .or_else(|| std::env::var("ORGONZOLA_LLM_GGUF").ok())
        .filter(|f| !f.is_empty());
    let dir = file.as_deref().and_then(|f| state.llm_dir_for(f));
    if !core_assistant::feature_enabled() || !settings.llm_enabled || dir.is_none() {
        let _ = AssistantChatEvent::phase(&run_id, "disabled").emit(&app);
        return Ok(());
    }
    let dir = dir.unwrap_or_default();
    let file = file.unwrap_or_default();
    let Some(board) = state
        .store
        .board(&board_id)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Err(format!("no board {board_id}"));
    };
    let repo_ids = state
        .store
        .board_effective_repo_ids(&board_id)
        .await
        .map_err(|e| e.to_string())?;

    // The first turn loads the model; tell the panel we are working while the agent runs tools.
    let _ = AssistantChatEvent::phase(&run_id, "thinking").emit(&app);

    let req = core_assistant::AssistantRequest {
        model_dir: dir,
        gguf_file: file,
        repo_ids,
        now: now_rfc3339(),
        board_label: board.name,
        active_tab: (!tab.is_empty()).then_some(tab),
        history: history
            .into_iter()
            .map(|t| core_assistant::ChatTurn {
                role: t.role,
                content: t.content,
            })
            .collect(),
        question,
    };
    let store = std::sync::Arc::new(state.store.clone());
    let embedder = state.embedder.clone();
    // Stream the agent's answer token by token. Tool calls surface as a "tool" phase (the panel
    // keeps its thinking state while a tool runs).
    let ev_app = app.clone();
    let ev_run = run_id.clone();
    // Whether any answer text was streamed. If the agent fails after some answer is on screen,
    // finish with "done" to keep it: "disabled" makes the UI replace the text.
    let streamed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ev_streamed = streamed.clone();
    let result = core_assistant::answer_stream(store, embedder, req, move |event| {
        let (phase, delta) = match event {
            core_assistant::AssistantEvent::Delta(text) => {
                ev_streamed.store(true, std::sync::atomic::Ordering::Relaxed);
                ("delta", text)
            }
            core_assistant::AssistantEvent::Tool(name) => ("tool", name),
            core_assistant::AssistantEvent::Done => return,
        };
        let _ = AssistantChatEvent {
            run_id: ev_run.clone(),
            phase: phase.to_string(),
            delta,
        }
        .emit(&ev_app);
    })
    .await;
    let _ = match result {
        Ok(()) => AssistantChatEvent::phase(&run_id, "done").emit(&app),
        // Failure with a partial answer on screen: end cleanly so it is not wiped. Log the cause.
        Err(e) if streamed.load(std::sync::atomic::Ordering::Relaxed) => {
            eprintln!("assistant chat error (after partial answer): {e}");
            AssistantChatEvent::phase(&run_id, "done").emit(&app)
        }
        // Failure before any text streamed. The `ready` gate already ruled out off/no-model, so
        // this is an engine error; surface its message.
        Err(e) => {
            eprintln!("assistant chat failed: {e}");
            AssistantChatEvent {
                run_id: run_id.clone(),
                phase: "error".to_string(),
                delta: e.to_string(),
            }
            .emit(&app)
        }
    };
    Ok(())
}

/// The attention signals a board can toggle: the watchdog `kind` plus a UI label, in display
/// order. A board with no disabled-signal rows flags all of them. `upstream` is computed
/// separately, so `board_overview` drops it explicitly when disabled (it never appears in
/// `board_attention`/`board_standup`/detail).
const ATTENTION_SIGNALS: [(&str, &str); 10] = [
    ("review_wait", "Waiting on first review"),
    ("stale_pr", "Stale PRs"),
    ("risky_change", "Risky change (touches dormant code)"),
    ("aging_wip", "Aging work in progress"),
    ("merged_without_review", "Merged without review"),
    ("failing_ci", "Failing CI"),
    ("flaky_ci", "Flaky CI (green only after a re-run)"),
    ("done_not_done", "Merged but issue still open"),
    ("orphan_pr", "Merged with no tracked issue"),
    ("upstream", "Upstream risk"),
];

/// One attention-signal preference for a board: signal id, label, and whether the board flags it.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SignalPrefView {
    pub signal: String,
    pub label: String,
    pub enabled: bool,
}

/// A board's attention-signal preferences: every toggleable signal and whether this board flags
/// it. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn board_signals(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<SignalPrefView>, String> {
    let disabled = state
        .store
        .board_disabled_signals(&id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(ATTENTION_SIGNALS
        .iter()
        .map(|(signal, label)| SignalPrefView {
            signal: (*signal).to_string(),
            label: (*label).to_string(),
            enabled: !disabled.iter().any(|d| d == signal),
        })
        .collect())
}

/// Enable or disable one attention signal for a board. Returns the updated preference list.
#[tauri::command]
#[specta::specta]
async fn set_board_signal_enabled(
    state: State<'_, AppState>,
    id: String,
    signal: String,
    enabled: bool,
) -> Result<Vec<SignalPrefView>, String> {
    if !ATTENTION_SIGNALS.iter().any(|(s, _)| *s == signal) {
        return Err(format!("unknown attention signal: {signal}"));
    }
    state
        .store
        .set_board_signal_enabled(&id, &signal, enabled)
        .await
        .map_err(|e| e.to_string())?;
    board_signals(state, id).await
}

/// A flagged pull request for the attention detail tables. Boundary form of
/// `core_summary::AttentionPr`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AttentionPrView {
    pub kind: String,
    pub id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
    pub created_at: String,
    pub merged_at: Option<String>,
    pub review_count: u32,
    pub approved: bool,
    pub url: Option<String>,
}

impl From<core_summary::AttentionPr> for AttentionPrView {
    fn from(p: core_summary::AttentionPr) -> Self {
        Self {
            kind: p.kind,
            id: p.id,
            number: p.number,
            title: p.title,
            state: p.state,
            author_login: p.author_login,
            created_at: p.created_at,
            merged_at: p.merged_at,
            review_count: p.review_count as u32,
            approved: p.approved,
            url: p.url,
        }
    }
}

/// A flagged CI run for the attention detail tables. Boundary form of `core_summary::AttentionCi`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AttentionCiView {
    pub id: String,
    pub commit_sha: Option<String>,
    pub status: String,
    pub conclusion: Option<String>,
    pub completed_at: Option<String>,
    pub url: Option<String>,
}

impl From<core_summary::AttentionCi> for AttentionCiView {
    fn from(c: core_summary::AttentionCi) -> Self {
        Self {
            id: c.id,
            commit_sha: c.commit_sha,
            status: c.status,
            conclusion: c.conclusion,
            completed_at: c.completed_at,
            url: c.url,
        }
    }
}

/// A repo's flagged items grouped by type, for the attention detail tables.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AttentionDetailView {
    pub prs: Vec<AttentionPrView>,
    pub ci: Vec<AttentionCiView>,
}

/// A board-wide pull request row for the Changes grid: boundary form of `core_summary::BoardPr`
/// plus the repo's `full_name`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardPrView {
    pub repo_id: String,
    pub full_name: String,
    pub id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
    pub created_at: String,
    pub merged_at: Option<String>,
    pub review_count: u32,
    pub approved: bool,
    pub url: Option<String>,
}

/// Every pull request across a board's effective repos, for the Changes grid. Sorted and filtered
/// client-side; a row's digest loads on click via `change_digest`. Read-only.
#[tauri::command]
#[specta::specta]
async fn board_changes(state: State<'_, AppState>, id: String) -> Result<Vec<BoardPrView>, String> {
    let repos = board_repos_resolved(&state.store, &id).await?;
    let names: std::collections::HashMap<String, String> = repos
        .iter()
        .map(|r| (r.id.clone(), r.full_name.clone()))
        .collect();
    let ids: Vec<String> = repos.into_iter().map(|r| r.id).collect();
    let prs = core_summary::board_pull_requests(&state.store, &ids)
        .await
        .map_err(|e| e.to_string())?;
    Ok(prs
        .into_iter()
        .map(|p| BoardPrView {
            full_name: names.get(&p.repo_id).cloned().unwrap_or_default(),
            repo_id: p.repo_id,
            id: p.id,
            number: p.number,
            title: p.title,
            state: p.state,
            author_login: p.author_login,
            created_at: p.created_at,
            merged_at: p.merged_at,
            review_count: p.review_count as u32,
            approved: p.approved,
            url: p.url,
        })
        .collect())
}

/// Cap on attention items one `board_overview` response carries. A large org board sent 7606
/// enriched items (4.0 MB) in one IPC response and the webview choked. Each repo still reports its
/// true `attention_total`; this bounds the payload, not what the signals find.
const MAX_BOARD_ATTENTION_ITEMS: usize = 200;

/// Keep the `cap` most severe attention items across every repo's list, in place, and return the
/// total before bounding. Severity order matches `core_summary::attention_severity_rank`; ties
/// keep their original (repo, position) order. Pure, so the bounding rule is unit-testable.
fn bound_board_attention(per_repo: &mut [Vec<AttentionView>], cap: usize) -> usize {
    let total: usize = per_repo.iter().map(Vec::len).sum();
    if total <= cap {
        return total;
    }
    let mut ranked: Vec<(usize, usize, usize)> = Vec::with_capacity(total);
    for (ri, items) in per_repo.iter().enumerate() {
        for (ii, a) in items.iter().enumerate() {
            ranked.push((ri, ii, core_summary::attention_severity_rank(&a.kind)));
        }
    }
    ranked.sort_by_key(|&(_, _, rank)| rank);
    let keep: std::collections::HashSet<(usize, usize)> = ranked
        .into_iter()
        .take(cap)
        .map(|(ri, ii, _)| (ri, ii))
        .collect();
    for (ri, items) in per_repo.iter_mut().enumerate() {
        let mut i = 0usize;
        items.retain(|_| {
            let keep_this = keep.contains(&(ri, i));
            i += 1;
            keep_this
        });
    }
    total
}

/// A per-repo overview line for a board: the repo plus its digest and attention count. Total
/// attention payload is bounded to `MAX_BOARD_ATTENTION_ITEMS`; each repo keeps its true
/// `attention_total` so the UI can say "showing N of M".
#[tauri::command]
#[specta::specta]
async fn board_overview(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<RepoOverview>, String> {
    let now = now_rfc3339();
    let repos = board_repos_resolved(&state.store, &id).await?;
    // Signals this board has turned off: their items are dropped, and so from the per-repo counts.
    let disabled = state
        .store
        .board_disabled_signals(&id)
        .await
        .map_err(|e| e.to_string())?;
    // Fork relationships (repo id -> parent's namespaced id), so the UI can fold observed
    // contributor forks out of the flagged list.
    let forks: std::collections::HashMap<String, Option<String>> = state
        .store
        .repo_forks()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|f| (f.repo_id, f.parent_repo_id))
        .collect();
    // Board-level WIP aging bands, computed once across all repos so each per-repo attention call
    // uses the same board-wide in-progress-time history.
    let repo_ids: Vec<String> = repos.iter().map(|r| r.id.clone()).collect();
    let wip_bands = core_summary::board_wip_aging_bands(&state.store, &repo_ids, &now)
        .await
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for repo in repos {
        let digest = core_summary::repo_digest(&state.store, &repo.id, &now)
            .await
            .map_err(|e| e.to_string())?;
        let attention = core_summary::attention_enriched(&state.store, &repo.id, &now, &wip_bands)
            .await
            .map_err(|e| e.to_string())?;
        // The two evidence-shortfall counts: merged PRs a signal could not judge because the store
        // never had the evidence. A disabled signal reports no shortfall.
        let coverage = core_summary::attention_coverage(&state.store, &repo.id)
            .await
            .map_err(|e| e.to_string())?;
        let merged_without_review_unjudged =
            if disabled.iter().any(|d| d == "merged_without_review") {
                0
            } else {
                coverage.merged_without_review_unjudged as u32
            };
        let orphan_pr_unjudged = if disabled.iter().any(|d| d == "orphan_pr") {
            0
        } else {
            coverage.orphan_pr_unjudged as u32
        };
        // The upstream alert is computed separately from the watchdog signals; when disabled,
        // skip the query and leave the alert list empty.
        let upstream = if disabled.iter().any(|d| d == "upstream") {
            Vec::new()
        } else {
            core_summary::upstream_alerts(&state.store, &repo.id, &now)
                .await
                .map_err(|e| e.to_string())?
        };
        let index_code = state
            .store
            .repo_index_code(&repo.id)
            .await
            .map_err(|e| e.to_string())?;
        let is_fork = forks.contains_key(&repo.id);
        let parent_full_name = forks
            .get(&repo.id)
            .and_then(|p| p.as_deref())
            .and_then(parent_full_name_of);
        let full_name = repo.full_name.clone();
        let attention: Vec<AttentionView> = attention
            .into_iter()
            .filter(|a| !disabled.contains(&a.kind))
            .map(|a| {
                let mut v = AttentionView::from(a);
                v.repo_full_name = Some(full_name.clone());
                v
            })
            .collect();
        out.push(RepoOverview {
            repo_id: repo.id,
            full_name: repo.full_name,
            ownership: repo.ownership,
            digest: digest.into(),
            attention_total: attention.len() as u32,
            attention,
            merged_without_review_unjudged,
            orphan_pr_unjudged,
            upstream: upstream.into_iter().map(UpstreamAlertView::from).collect(),
            index_code,
            is_fork,
            parent_full_name,
        });
    }
    let mut lists: Vec<Vec<AttentionView>> = out
        .iter_mut()
        .map(|r| std::mem::take(&mut r.attention))
        .collect();
    bound_board_attention(&mut lists, MAX_BOARD_ATTENTION_ITEMS);
    for (r, items) in out.iter_mut().zip(lists) {
        r.attention = items;
    }
    Ok(out)
}

/// The `owner/name` inside a namespaced repo id (`repo:<forge>/owner/name`), for showing a
/// fork's upstream. `None` if the id is not in that shape.
fn parent_full_name_of(repo_id: &str) -> Option<String> {
    repo_id
        .strip_prefix("repo:")?
        .split_once('/')
        .map(|(_forge, full)| full.to_string())
}

/// DORA-lite metrics. Boundary form of `core_summary::DoraMetrics`. Deploy frequency is measured;
/// lead time and change failure rate are proxies (the UI marks them).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DoraView {
    pub window_days: i64,
    pub deploy_frequency_per_week: Option<f64>,
    pub deploy_tier: String,
    pub lead_time_secs: Option<i64>,
    pub lead_tier: String,
    /// True when lead time is the deploy-based value, false when it is the cycle-time proxy.
    pub lead_from_deploys: bool,
    pub change_failure_rate: Option<f64>,
    pub cfr_tier: String,
}

impl From<core_summary::DoraMetrics> for DoraView {
    fn from(d: core_summary::DoraMetrics) -> Self {
        Self {
            window_days: d.window_days,
            deploy_frequency_per_week: d.deploy_frequency_per_week,
            deploy_tier: d.deploy_tier,
            lead_time_secs: d.lead_time_secs,
            lead_tier: d.lead_tier,
            lead_from_deploys: d.lead_from_deploys,
            change_failure_rate: d.change_failure_rate,
            cfr_tier: d.cfr_tier,
        }
    }
}

/// DORA-lite metrics for a board, pooled across its repos.
#[tauri::command]
#[specta::specta]
async fn board_dora(state: State<'_, AppState>, id: String) -> Result<DoraView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let d = core_summary::board_dora(
        &state.store,
        &repo_ids,
        &now_rfc3339(),
        core_summary::DORA_WINDOW_DAYS,
    )
    .await
    .map_err(|e| e.to_string())?;
    Ok(d.into())
}

/// A code hotspot: a file that changes often and churns a lot.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct HotspotView {
    pub repo_id: String,
    pub full_name: String,
    pub path: String,
    pub changes: i64,
    pub churn: i64,
    pub authors: i64,
    pub score: f64,
}

/// The hottest files across a board's repos: change frequency x churn, generated/vendored paths
/// excluded. The Code tab renders these as a treemap.
#[tauri::command]
#[specta::specta]
async fn board_hotspots(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<HotspotView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let hotspots = core_summary::board_hotspots(&state.store, &repo_ids, 40)
        .await
        .map_err(|e| e.to_string())?;
    Ok(hotspots
        .into_iter()
        .map(|h| HotspotView {
            full_name: full_name_of_repo_id(&h.repo_id),
            repo_id: h.repo_id,
            path: h.path,
            changes: h.changes,
            churn: h.churn,
            authors: h.authors,
            score: h.score,
        })
        .collect())
}

/// A PR that touched a code-risk file, for the drill-down.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CodeRiskPrView {
    pub number: i64,
    pub title: String,
    pub url: Option<String>,
}

/// A file's code risk: hotspot x ownership x review, with the components, plain-language reasons
/// and the recent PRs that touched it. Code risk, not a people ranking.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CodeRiskView {
    pub repo_id: String,
    pub full_name: String,
    pub path: String,
    pub changes: i64,
    pub churn: i64,
    pub authors: i64,
    pub top_author: String,
    pub top_share: f64,
    pub bus_factor: i64,
    pub merged_touches: i64,
    pub unreviewed: i64,
    pub review_gap: f64,
    pub score: f64,
    pub reasons: Vec<String>,
    pub recent_prs: Vec<CodeRiskPrView>,
}

/// The riskiest files across a board's repos: hotspot x ownership x review per file, with
/// drill-to-PR. The Code tab renders these as an explainable risk list.
#[tauri::command]
#[specta::specta]
async fn board_code_risk(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<CodeRiskView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let risks = core_summary::board_code_risk(&state.store, &repo_ids, 40)
        .await
        .map_err(|e| e.to_string())?;
    Ok(risks
        .into_iter()
        .map(|r| CodeRiskView {
            full_name: full_name_of_repo_id(&r.repo_id),
            repo_id: r.repo_id,
            path: r.path,
            changes: r.changes,
            churn: r.churn,
            authors: r.authors,
            top_author: r.top_author,
            top_share: r.top_share,
            bus_factor: r.bus_factor,
            merged_touches: r.merged_touches,
            unreviewed: r.unreviewed,
            review_gap: r.review_gap,
            score: r.score,
            reasons: r.reasons,
            recent_prs: r
                .recent_prs
                .into_iter()
                .map(|p| CodeRiskPrView {
                    number: p.number,
                    title: p.title,
                    url: p.url,
                })
                .collect(),
        })
        .collect())
}

/// A file's AST health score for the UI Code tab.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct FileHealthView {
    pub repo_id: String,
    pub full_name: String,
    pub path: String,
    pub loc: i64,
    pub functions: i64,
    pub branches: i64,
    /// Health score 1-10 (10 = simplest), from branch density and LOC.
    pub score: i64,
}

/// The weakest files across a board's repos by AST health score, lowest first, capped at 40. Only
/// parsed files are scored, so a file in a language with no tree-sitter grammar is absent, not
/// healthy. Empty when code indexing has not run.
#[tauri::command]
#[specta::specta]
async fn board_code_health(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<FileHealthView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let items = core_summary::board_code_health(&state.store, &repo_ids, 40)
        .await
        .map_err(|e| e.to_string())?;
    Ok(items
        .into_iter()
        .map(|h| FileHealthView {
            full_name: full_name_of_repo_id(&h.repo_id),
            repo_id: h.repo_id,
            path: h.path,
            loc: h.loc,
            functions: h.functions,
            branches: h.branches,
            score: h.score,
        })
        .collect())
}

/// One scorecard rule's outcome. `status` is "pass" | "fail" | "unknown"; `unknown` means the
/// underlying fact was never observed, which is not a `fail`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RuleResultView {
    pub rule: String,
    pub tier: String,
    pub status: String,
    pub detail: String,
}

/// A repo's scorecard for the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ScorecardView {
    pub repo_id: String,
    pub full_name: String,
    pub tier: String,
    pub rules: Vec<RuleResultView>,
}

/// A board's scorecard composite for the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardScorecardView {
    pub composite_tier: String,
    pub gold: u32,
    pub silver: u32,
    pub bronze: u32,
    pub none: u32,
    pub repos: Vec<ScorecardView>,
}

/// A board's work-item link coverage: the share of merged PRs that resolve to a work-item link,
/// with the authoritative `closes` count split out.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct LinkCoverageView {
    pub merged_total: u32,
    pub linked: u32,
    pub closes: u32,
}

#[tauri::command]
#[specta::specta]
async fn board_link_coverage(
    state: State<'_, AppState>,
    id: String,
) -> Result<LinkCoverageView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let cov = core_summary::board_link_coverage(&state.store, &repo_ids)
        .await
        .map_err(|e| e.to_string())?;
    Ok(LinkCoverageView {
        merged_total: cov.merged_total as u32,
        linked: cov.linked as u32,
        closes: cov.closes as u32,
    })
}

/// A board's work-item flow: lead time and WIP from the derived status history. Empty on a board
/// with no linked GitHub issues. Read-only; works with no token.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct FlowMetricsView {
    pub completed: u32,
    pub median_lead_time_secs: Option<i64>,
    pub median_wait_secs: Option<i64>,
    pub median_active_secs: Option<i64>,
    pub in_progress: u32,
    pub median_in_progress_age_secs: Option<i64>,
}

#[tauri::command]
#[specta::specta]
async fn board_flow_metrics(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<FlowMetricsView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let (since, until) = window_bounds(since_days, until_days);
    let m =
        core_summary::board_flow_metrics(&state.store, &repo_ids, &since, &until, &now_rfc3339())
            .await
            .map_err(|e| e.to_string())?;
    Ok(FlowMetricsView {
        completed: m.completed as u32,
        median_lead_time_secs: m.median_lead_time_secs,
        median_wait_secs: m.median_wait_secs,
        median_active_secs: m.median_active_secs,
        in_progress: m.in_progress as u32,
        median_in_progress_age_secs: m.median_in_progress_age_secs,
    })
}

/// A board's PR delivery-cycle phase medians: pickup (open -> first review), review (-> merge).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardCyclePhasesView {
    pub median_pickup_secs: Option<i64>,
    pub median_review_secs: Option<i64>,
}

#[tauri::command]
#[specta::specta]
async fn board_cycle_phases(
    state: State<'_, AppState>,
    id: String,
) -> Result<BoardCyclePhasesView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let p = core_summary::board_cycle_phases(&state.store, &repo_ids)
        .await
        .map_err(|e| e.to_string())?;
    Ok(BoardCyclePhasesView {
        median_pickup_secs: p.median_pickup_secs,
        median_review_secs: p.median_review_secs,
    })
}

/// One day of a board's flow/quality trend. Boundary form of `core_summary::FlowTrendPoint`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct FlowTrendView {
    pub captured_on: String,
    pub lead_time_secs: Option<i64>,
    pub wip: i64,
    pub bug_net: i64,
}

#[tauri::command]
#[specta::specta]
async fn board_flow_trend(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<FlowTrendView>, String> {
    let points = core_summary::board_flow_trend(&state.store, &id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(points
        .into_iter()
        .map(|p| FlowTrendView {
            captured_on: p.captured_on,
            lead_time_secs: p.lead_time_secs,
            wip: p.wip,
            bug_net: p.bug_net,
        })
        .collect())
}

/// A board's bug inflow vs outflow over a window: bug-labeled issues opened vs closed.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BugFlowView {
    pub opened: u32,
    pub closed: u32,
}

#[tauri::command]
#[specta::specta]
async fn board_bug_flow(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<BugFlowView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let (since, until) = window_bounds(since_days, until_days);
    let f = core_summary::board_bug_flow(&state.store, &repo_ids, &since, &until)
        .await
        .map_err(|e| e.to_string())?;
    Ok(BugFlowView {
        opened: f.opened as u32,
        closed: f.closed as u32,
    })
}

/// A sprint's say-do: committed vs delivered plus scope churn. `ratio` is null when nothing was
/// committed.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SayDoView {
    pub sprint_id: String,
    pub sprint_name: String,
    pub state: Option<String>,
    pub committed_on: Option<String>,
    pub committed: u32,
    pub delivered: u32,
    pub carryover: u32,
    pub added: u32,
    pub removed: u32,
    pub ratio: Option<f64>,
}

/// Say-do for every Jira sprint a board watches, newest first. Only sprints with a frozen
/// commitment.
#[tauri::command]
#[specta::specta]
async fn board_say_do(state: State<'_, AppState>, id: String) -> Result<Vec<SayDoView>, String> {
    let rows = core_summary::board_say_do(&state.store, &id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .filter(|s| s.committed_on.is_some())
        .map(|s| SayDoView {
            sprint_id: s.sprint_id,
            sprint_name: s.sprint_name,
            state: s.state,
            committed_on: s.committed_on,
            committed: s.committed as u32,
            delivered: s.delivered as u32,
            carryover: s.carryover as u32,
            added: s.added as u32,
            removed: s.removed as u32,
            ratio: s.ratio,
        })
        .collect())
}

/// One point of a board's say-do velocity trend: a sprint with a frozen commitment and a
/// measurable ratio. Carries the counts the ratio is made of, so 1-of-1 is not read as 20-of-20.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct VelocityPointView {
    pub sprint_id: String,
    pub sprint_name: String,
    pub committed_on: String,
    pub committed: u32,
    pub delivered: u32,
    pub ratio: f64,
}

/// A board's say-do velocity trend, oldest sprint first. The order comes from core
/// (`velocity_trend`, by commitment date); the UI plots it as given. `filter_map` only unwraps
/// fields core guarantees, so the view has no nullable ratio.
#[tauri::command]
#[specta::specta]
async fn board_velocity_trend(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<VelocityPointView>, String> {
    let rows = core_summary::board_say_do(&state.store, &id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(core_summary::velocity_trend(&rows)
        .into_iter()
        .filter_map(|s| {
            Some(VelocityPointView {
                sprint_id: s.sprint_id.clone(),
                sprint_name: s.sprint_name.clone(),
                committed_on: s.committed_on.clone()?,
                committed: s.committed as u32,
                delivered: s.delivered as u32,
                ratio: s.ratio?,
            })
        })
        .collect())
}

/// A board's scorecards: per-repo Bronze/Silver/Gold plus a composite (the weakest repo's tier).
/// Composes flow, CI and ownership signals; rules-before-LLM, no people ranking.
#[tauri::command]
#[specta::specta]
async fn board_scorecard(
    state: State<'_, AppState>,
    id: String,
) -> Result<BoardScorecardView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let sc = core_summary::board_scorecard(&state.store, &repo_ids, &now_rfc3339())
        .await
        .map_err(|e| e.to_string())?;
    let map_rules = |rules: Vec<core_scorecard::RuleResult>| -> Vec<RuleResultView> {
        rules
            .into_iter()
            .map(|r| RuleResultView {
                rule: r.rule,
                tier: r.tier,
                status: r.status.as_str().to_string(),
                detail: r.detail,
            })
            .collect()
    };
    Ok(BoardScorecardView {
        composite_tier: sc.composite_tier,
        gold: sc.gold as u32,
        silver: sc.silver as u32,
        bronze: sc.bronze as u32,
        none: sc.none as u32,
        repos: sc
            .repos
            .into_iter()
            .map(|r| ScorecardView {
                full_name: full_name_of_repo_id(&r.repo_id),
                repo_id: r.repo_id,
                tier: r.tier,
                rules: map_rules(r.rules),
            })
            .collect(),
    })
}

/// A dependency used across a board: the repos (by full name) that declare it.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DepUsageView {
    pub ecosystem: String,
    pub name: String,
    pub repos: Vec<String>,
}

/// The dependencies across a board's repos, each with the repos that use it; dependencies shared
/// by more than one repo lead. The Code tab renders a list and graph.
#[tauri::command]
#[specta::specta]
async fn board_dependencies(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<DepUsageView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let deps = core_summary::board_dependencies(&state.store, &repo_ids)
        .await
        .map_err(|e| e.to_string())?;
    Ok(deps
        .into_iter()
        .map(|d| DepUsageView {
            ecosystem: d.ecosystem,
            name: d.name,
            repos: d.repos.iter().map(|r| full_name_of_repo_id(r)).collect(),
        })
        .collect())
}

/// A dependency package with known advisories. Package-level: manifests declare ranges, not
/// resolved versions.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PackageAdvisoryView {
    pub ecosystem: String,
    pub name: String,
    pub advisories: Vec<String>,
}

/// Toggle opt-in dependency vulnerability scanning for a board. Returns the updated board.
#[tauri::command]
#[specta::specta]
async fn set_board_scan_dependencies(
    state: State<'_, AppState>,
    id: String,
    enabled: bool,
) -> Result<BoardView, String> {
    state
        .store
        .set_board_scan_dependencies(&id, enabled)
        .await
        .map_err(|e| e.to_string())?;
    require_board(&state.store, &id).await
}

/// Scan a board's dependency packages against the OSV advisory database. Network egress; only
/// runs when the board has scanning enabled. Package-level (version-imprecise).
#[tauri::command]
#[specta::specta]
async fn scan_board_dependencies(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<PackageAdvisoryView>, String> {
    let board = state
        .store
        .board(&id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no board {id}"))?;
    if !board.scan_dependencies {
        return Err("dependency scanning is off for this board; enable it in Settings".into());
    }
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let deps = core_summary::board_dependencies(&state.store, &repo_ids)
        .await
        .map_err(|e| e.to_string())?;
    let packages: Vec<(String, String)> = deps.into_iter().map(|d| (d.ecosystem, d.name)).collect();
    let hits = osv::query_osv(osv::OSV_BATCH_URL, &packages).await?;
    Ok(hits
        .into_iter()
        .map(|h| PackageAdvisoryView {
            ecosystem: h.ecosystem,
            name: h.name,
            advisories: h.advisories,
        })
        .collect())
}

/// A change-coupling pair: two files in the same repo that often change in the same merged PR.
/// Implicit coupling not visible in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CouplingPairView {
    pub repo_id: String,
    pub full_name: String,
    pub path_a: String,
    pub path_b: String,
    /// Merged PRs where both files changed together.
    pub together: i64,
    /// Jaccard coupling strength: `together / (prs_a + prs_b - together)`.
    pub coupling: f64,
}

/// Top change-coupling pairs across a board's repos. Shown on the Code tab beside hotspots.
#[tauri::command]
#[specta::specta]
async fn board_coupling(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<CouplingPairView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let pairs = core_summary::board_coupling(&state.store, &repo_ids, 20)
        .await
        .map_err(|e| e.to_string())?;
    Ok(pairs
        .into_iter()
        .map(|p| CouplingPairView {
            full_name: full_name_of_repo_id(&p.repo_id),
            repo_id: p.repo_id,
            path_a: p.path_a,
            path_b: p.path_b,
            together: p.together,
            coupling: p.coupling,
        })
        .collect())
}

/// A module's ownership risk for the UI. Code risk, not a people ranking.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct OwnershipRiskView {
    pub repo_id: String,
    pub full_name: String,
    pub module: String,
    pub changes: i64,
    pub authors: i64,
    pub top_author: String,
    pub top_share: f64,
    pub bus_factor: i64,
}

/// At-risk modules across a board's repos: low bus factor or high concentration ("module X has
/// bus factor N, top owner <login> at S%"). Framed as code risk, never a ranking of people.
#[tauri::command]
#[specta::specta]
async fn board_ownership_risks(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<OwnershipRiskView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let risks = core_summary::board_ownership_risks(&state.store, &repo_ids, 20)
        .await
        .map_err(|e| e.to_string())?;
    Ok(risks
        .into_iter()
        .map(|r| OwnershipRiskView {
            full_name: full_name_of_repo_id(&r.repo_id),
            repo_id: r.repo_id,
            module: r.module,
            changes: r.changes,
            authors: r.authors,
            top_author: r.top_author,
            top_share: r.top_share,
            bus_factor: r.bus_factor,
        })
        .collect())
}

/// A tracked issue on the standup. Boundary form of `core_ritual::IssueRef`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct IssueRefView {
    pub id: String,
    pub number: i64,
    pub title: String,
}

impl From<core_ritual::IssueRef> for IssueRefView {
    fn from(i: core_ritual::IssueRef) -> Self {
        Self {
            id: i.id,
            number: i.number,
            title: i.title,
        }
    }
}

/// A board's standup agenda, rolled up across the board's repos.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardStandupView {
    pub board_id: String,
    pub since: String,
    pub until: String,
    pub moved_commits: u32,
    pub merged_prs: Vec<PrRefView>,
    pub waiting_on_review: Vec<PrRefView>,
    /// Issues delivered (closed) in the window, and open issues in progress.
    pub delivered_issues: Vec<IssueRefView>,
    pub in_progress_issues: Vec<IssueRefView>,
    pub needs_attention: Vec<AttentionView>,
}

/// The standup agenda for a whole board over a window: what moved within `since_days..until_days`
/// ago (day offsets from now; `until_days` 0 means now, so "this week" is 7..0 and "last week"
/// is 14..7), what waits on review, and what needs attention. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn board_standup(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<BoardStandupView, String> {
    // `since` is the older edge; clamp and order so the window is never inverted.
    let lo = since_days.clamp(1, 365);
    let hi = until_days.clamp(0, 365).min(lo);
    let since = days_ago_rfc3339(lo);
    let until = days_ago_rfc3339(hi);
    let now = now_rfc3339();
    let repos = board_repos_resolved(&state.store, &id).await?;
    // The standup respects the board's disabled signals, as the Attention tab does.
    let disabled = state
        .store
        .board_disabled_signals(&id)
        .await
        .map_err(|e| e.to_string())?;
    // Observed contributor forks are folded out: their fork-side activity (a PR-staging fork's
    // failing CI) is not the team's work. A pinned (owned) fork is kept.
    let fork_ids: std::collections::HashSet<String> = state
        .store
        .repo_forks()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|f| f.repo_id)
        .collect();
    let mut moved_commits = 0u32;
    let mut merged_prs = Vec::new();
    let mut waiting_on_review = Vec::new();
    let mut delivered_issues = Vec::new();
    let mut in_progress_issues = Vec::new();
    let mut needs_attention = Vec::new();
    // Board-level WIP aging bands once, so aging items carry the same badge as the Attention tab.
    let standup_repo_ids: Vec<String> = repos.iter().map(|r| r.id.clone()).collect();
    let wip_bands = core_summary::board_wip_aging_bands(&state.store, &standup_repo_ids, &now)
        .await
        .map_err(|e| e.to_string())?;
    for repo in repos {
        if repo.ownership == "observed" && fork_ids.contains(&repo.id) {
            continue;
        }
        let full_name = repo.full_name.clone();
        let agenda = core_ritual::standup(&state.store, &repo.id, &since, &until, &now, &wip_bands)
            .await
            .map_err(|e| e.to_string())?;
        moved_commits += agenda.moved_commits as u32;
        merged_prs.extend(agenda.merged_prs.into_iter().map(PrRefView::from));
        waiting_on_review.extend(agenda.waiting_on_review.into_iter().map(PrRefView::from));
        delivered_issues.extend(agenda.delivered_issues.into_iter().map(IssueRefView::from));
        in_progress_issues.extend(
            agenda
                .in_progress_issues
                .into_iter()
                .map(IssueRefView::from),
        );
        needs_attention.extend(
            agenda
                .needs_attention
                .into_iter()
                .filter(|a| !disabled.contains(&a.kind))
                .map(|a| {
                    let mut v = AttentionView::from(a);
                    v.repo_full_name = Some(full_name.clone());
                    v
                }),
        );
    }
    Ok(BoardStandupView {
        board_id: id,
        since,
        until,
        moved_commits,
        merged_prs,
        waiting_on_review,
        delivered_issues,
        in_progress_issues,
        needs_attention,
    })
}

/// A board's standup as a shareable markdown digest: the `board_standup` roll-up (observed forks
/// skipped, disabled signals dropped), rendered per repo with each item's next-step action.
/// Deterministic, LLM-free, no egress. Read-only; works with no token.
async fn build_board_digest(
    store: &core_store::Store,
    id: &str,
    since_days: i64,
    until_days: i64,
) -> Result<String, String> {
    let lo = since_days.clamp(1, 365);
    let hi = until_days.clamp(0, 365).min(lo);
    let since = days_ago_rfc3339(lo);
    let until = days_ago_rfc3339(hi);
    let now = now_rfc3339();
    let Some(board) = store.board(id).await.map_err(|e| e.to_string())? else {
        return Err(format!("no board {id}"));
    };
    let repos = board_repos_resolved(store, id).await?;
    let disabled = store
        .board_disabled_signals(id)
        .await
        .map_err(|e| e.to_string())?;
    let fork_ids: std::collections::HashSet<String> = store
        .repo_forks()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|f| f.repo_id)
        .collect();
    // Board-level WIP aging bands once, the same baseline as the Attention tab.
    let digest_repo_ids: Vec<String> = repos.iter().map(|r| r.id.clone()).collect();
    let wip_bands = core_summary::board_wip_aging_bands(store, &digest_repo_ids, &now)
        .await
        .map_err(|e| e.to_string())?;
    let mut out = format!("# {} - standup {} to {}\n\n", board.name, since, until);
    for repo in repos {
        if repo.ownership == "observed" && fork_ids.contains(&repo.id) {
            continue;
        }
        let mut agenda = core_ritual::standup(store, &repo.id, &since, &until, &now, &wip_bands)
            .await
            .map_err(|e| e.to_string())?;
        agenda
            .needs_attention
            .retain(|a| !disabled.contains(&a.kind));
        out.push_str(&core_ritual::render_agenda_markdown(
            &agenda,
            &repo.full_name,
        ));
    }
    Ok(out)
}

#[tauri::command]
#[specta::specta]
async fn board_digest(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<String, String> {
    build_board_digest(&state.store, &id, since_days, until_days).await
}

/// Build and POST a board's standup digest to the configured Slack-compatible incoming webhook.
/// Egress happens only here, and only when a webhook URL is set. POSTs `{"text": <md>}`; returns
/// the digest sent. Public so the scheduled-digest task can call it.
pub async fn push_board_digest(
    store: &core_store::Store,
    id: &str,
    since_days: i64,
    until_days: i64,
) -> Result<String, String> {
    let settings = store.settings().await.map_err(|e| e.to_string())?;
    let Some(url) = settings.digest_webhook_url.filter(|u| !u.trim().is_empty()) else {
        return Err("no digest webhook configured (set one in Settings)".into());
    };
    let digest = build_board_digest(store, id, since_days, until_days).await?;
    let client = reqwest::Client::builder()
        .user_agent("orgonzola")
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "text": digest }))
        .send()
        .await
        .map_err(|e| format!("digest webhook request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("digest webhook returned {}", resp.status()));
    }
    Ok(digest)
}

/// Push a board's standup digest to the configured webhook on demand. Egress only on this call
/// with a configured URL.
#[tauri::command]
#[specta::specta]
async fn send_board_digest(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<String, String> {
    push_board_digest(&state.store, &id, since_days, until_days).await
}

/// One of a person's open PRs, for the per-person view. `repo` is `owner/name`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PersonPrView {
    pub repo: String,
    pub number: i64,
    pub title: String,
}

/// A person's activity across a board's repos. Boundary form of `core_summary::PersonActivity`.
/// Descriptive, no ranking.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PersonActivityView {
    pub login: String,
    pub open_prs: Vec<PersonPrView>,
    pub merged_prs: u32,
    pub reviews_given: u32,
    pub commits: u32,
    /// `owner/name` of each board repo the person appears in.
    pub repos: Vec<String>,
}

fn strip_repo(id: &str) -> String {
    id.strip_prefix("repo:").unwrap_or(id).to_string()
}

impl From<core_summary::PersonActivity> for PersonActivityView {
    fn from(a: core_summary::PersonActivity) -> Self {
        Self {
            login: a.login,
            open_prs: a
                .open_prs
                .into_iter()
                .map(|p| PersonPrView {
                    repo: strip_repo(&p.repo_id),
                    number: p.number,
                    title: p.title,
                })
                .collect(),
            merged_prs: a.merged_prs as u32,
            reviews_given: a.reviews_given as u32,
            commits: a.commits as u32,
            repos: a.repos.iter().map(|r| strip_repo(r)).collect(),
        }
    }
}

/// One person's activity across a board's effective repos: open PRs and merged-PR, review and
/// commit counts. Read-only over the synced store; works with no token.
#[tauri::command]
#[specta::specta]
async fn person_activity(
    state: State<'_, AppState>,
    id: String,
    login: String,
) -> Result<PersonActivityView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let activity = core_summary::person_activity(&state.store, &repo_ids, login.trim())
        .await
        .map_err(|e| e.to_string())?;
    Ok(activity.into())
}

/// One teammate's comparison stats for the People table. Boundary form of
/// `core_summary::PersonStats`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PersonStatsView {
    pub login: String,
    pub commits: u32,
    pub prs_opened: u32,
    pub prs_merged: u32,
    pub reviews_given: u32,
    pub open_prs: u32,
    pub approvals_given: u32,
    pub self_merges: u32,
    /// Median PR cycle time (open -> merge) in seconds, or null if nothing merged in the window.
    pub median_cycle_time_secs: Option<i64>,
    /// Distinct UTC days with a work event (consistency/cadence).
    pub active_days: u32,
    /// Median PR-open -> their-first-review latency in seconds, or null if nothing datable.
    pub median_review_latency_secs: Option<i64>,
    /// Average change size (additions + deletions) over merged PRs with synced files, or null.
    pub avg_pr_churn: Option<i64>,
    /// Issues closed by their merged PRs (authoritative links).
    pub issues_closed: u32,
    pub ci_failures: u32,
    pub off_hours_events: u32,
    pub total_events: u32,
}

impl From<core_summary::PersonStats> for PersonStatsView {
    fn from(s: core_summary::PersonStats) -> Self {
        Self {
            login: s.login,
            commits: s.commits as u32,
            prs_opened: s.prs_opened as u32,
            prs_merged: s.prs_merged as u32,
            reviews_given: s.reviews_given as u32,
            open_prs: s.open_prs as u32,
            approvals_given: s.approvals_given as u32,
            self_merges: s.self_merges as u32,
            median_cycle_time_secs: s.median_cycle_time_secs,
            active_days: s.active_days as u32,
            median_review_latency_secs: s.median_review_latency_secs,
            avg_pr_churn: s.avg_pr_churn,
            issues_closed: s.issues_closed as u32,
            ci_failures: s.ci_failures as u32,
            off_hours_events: s.off_hours_events as u32,
            total_events: s.total_events as u32,
        }
    }
}

/// The board People comparison: per-person throughput rows plus one team-level off-hours
/// sustainability counter, never attributed to an individual. Boundary form of
/// `core_summary::BoardPeopleStats`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BoardPeopleStatsView {
    pub people: Vec<PersonStatsView>,
    pub off_hours_events: u32,
    pub total_events: u32,
}

impl From<core_summary::BoardPeopleStats> for BoardPeopleStatsView {
    fn from(s: core_summary::BoardPeopleStats) -> Self {
        Self {
            people: s.people.into_iter().map(PersonStatsView::from).collect(),
            off_hours_events: s.off_hours_events as u32,
            total_events: s.total_events as u32,
        }
    }
}

/// Resolve a `since_days..until_days` window (day offsets from now, `until_days` 0 = now) into
/// RFC-3339 bounds, ordered so the window is never inverted. Shared by the People-stats commands.
fn window_bounds(since_days: i64, until_days: i64) -> (String, String) {
    let lo = since_days.clamp(1, 3650);
    let hi = until_days.clamp(0, 3650).min(lo);
    (days_ago_rfc3339(lo), days_ago_rfc3339(hi))
}

/// Comparison stats for every person on a board over a window: per-person throughput and current
/// load, plus one team-level off-hours counter (never per person). Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn board_people_stats(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<BoardPeopleStatsView, String> {
    let Some(board) = state.store.board(&id).await.map_err(|e| e.to_string())? else {
        return Err(format!("no board {id}"));
    };
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let (since, until) = window_bounds(since_days, until_days);
    // A team board watches a chosen roster; a repo/org board has none, so discover people from the
    // repos' activity in the window. Same stats path either way.
    let roster = if board.people.is_empty() {
        core_summary::contributors(&state.store, &repo_ids, &since, &until)
            .await
            .map_err(|e| e.to_string())?
    } else {
        board.people
    };
    let stats = core_summary::board_people_stats(&state.store, &repo_ids, &roster, &since, &until)
        .await
        .map_err(|e| e.to_string())?;
    Ok(BoardPeopleStatsView::from(stats))
}

/// One area a person has context in, for the pickup panel.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PersonAreaView {
    pub area: String,
    pub changes: i64,
}

/// A person's pickup profile: in-flight load, last-active time and the areas they have context
/// in. Descriptive, not a score or ranking.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PersonPickupView {
    pub login: String,
    pub open_prs: u32,
    pub last_active: Option<String>,
    pub areas: Vec<PersonAreaView>,
}

/// How many areas of familiarity to show per person in the pickup panel.
const PICKUP_AREAS: usize = 4;

/// The board's "who can pick this up" panel: per person, in-flight load, last-active time and
/// areas of context. Descriptive, not a ranking. The roster is the board's people, or
/// contributors discovered from activity, in a stable order. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn board_pickup(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<Vec<PersonPickupView>, String> {
    let Some(board) = state.store.board(&id).await.map_err(|e| e.to_string())? else {
        return Err(format!("no board {id}"));
    };
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let (since, until) = window_bounds(since_days, until_days);
    let roster = if board.people.is_empty() {
        core_summary::contributors(&state.store, &repo_ids, &since, &until)
            .await
            .map_err(|e| e.to_string())?
    } else {
        board.people
    };
    let rows = core_summary::board_pickup(&state.store, &repo_ids, &roster, PICKUP_AREAS)
        .await
        .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|p| PersonPickupView {
            login: p.login,
            open_prs: p.open_prs as u32,
            last_active: p.last_active,
            areas: p
                .areas
                .into_iter()
                .map(|a| PersonAreaView {
                    area: a.area,
                    changes: a.changes,
                })
                .collect(),
        })
        .collect())
}

/// One investment category's share: merged-PR count (primary) and churn (secondary).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct InvestmentBucketView {
    pub category: String,
    pub count: u32,
    pub churn: i64,
}

/// Where a board's delivered effort went over a window: merged PRs grouped into investment
/// categories. Board-level, not per person.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct InvestmentDistributionView {
    pub buckets: Vec<InvestmentBucketView>,
    pub total_count: u32,
    pub total_churn: i64,
}

/// The board's investment distribution over a window: merged PRs classified into feature / bug /
/// maintenance / docs / test / other by conventional-commit title and linked bug work items, by
/// count and churn. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn board_investment(
    state: State<'_, AppState>,
    id: String,
    since_days: i64,
    until_days: i64,
) -> Result<InvestmentDistributionView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let (since, until) = window_bounds(since_days, until_days);
    let dist = core_summary::board_investment(&state.store, &repo_ids, &since, &until)
        .await
        .map_err(|e| e.to_string())?;
    Ok(InvestmentDistributionView {
        buckets: dist
            .buckets
            .into_iter()
            .map(|b| InvestmentBucketView {
                category: b.category,
                count: b.count as u32,
                churn: b.churn,
            })
            .collect(),
        total_count: dist.total_count as u32,
        total_churn: dist.total_churn,
    })
}

/// One epic/initiative's progress: children done / total / in-progress plus an at-risk flag.
/// Child-count based, board-level.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct EpicProgressView {
    pub key: String,
    pub title: String,
    pub total: u32,
    pub done: u32,
    pub in_progress: u32,
    pub at_risk: bool,
}

/// A board's epic/initiative progress: Jira issues rolled up to their parent epics, with a
/// stalled-work at-risk flag. Empty when the board has no Jira parents. Read-only; no token.
#[tauri::command]
#[specta::specta]
async fn board_epics(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<EpicProgressView>, String> {
    let epics = core_summary::board_epics(&state.store, &id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(epics
        .into_iter()
        .map(|e| EpicProgressView {
            key: e.key,
            title: e.title,
            total: e.total as u32,
            done: e.done as u32,
            in_progress: e.in_progress as u32,
            at_risk: e.at_risk,
        })
        .collect())
}

/// When a person works: a 7x24 (weekday x hour, UTC) activity heatmap over a window,
/// `buckets[weekday*24 + hour]`, weekday 0 = Sunday. Read-only; works with no token.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct WorkPatternView {
    pub buckets: Vec<u32>,
}

#[tauri::command]
#[specta::specta]
async fn person_work_pattern(
    state: State<'_, AppState>,
    id: String,
    login: String,
    since_days: i64,
    until_days: i64,
) -> Result<WorkPatternView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let (since, until) = window_bounds(since_days, until_days);
    let pattern =
        core_summary::person_work_pattern(&state.store, &repo_ids, login.trim(), &since, &until)
            .await
            .map_err(|e| e.to_string())?;
    Ok(WorkPatternView {
        buckets: pattern.buckets,
    })
}

/// One day on a board's metric trend. Boundary form of `core_summary::TrendPoint`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct TrendPointView {
    pub captured_on: String,
    pub wip: i64,
    pub stale_open_prs: i64,
    pub merged_without_review: i64,
    pub attention_count: i64,
}

impl From<core_summary::TrendPoint> for TrendPointView {
    fn from(p: core_summary::TrendPoint) -> Self {
        Self {
            captured_on: p.captured_on,
            wip: p.wip,
            stale_open_prs: p.stale_open_prs,
            merged_without_review: p.merged_without_review,
            attention_count: p.attention_count,
        }
    }
}

/// A board's metric trend: per day, totals across the board's effective repos. Read-only over
/// recorded snapshots; works with no token. Empty until snapshots accumulate.
#[tauri::command]
#[specta::specta]
async fn board_trend(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<TrendPointView>, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let trend = core_summary::board_trend(&state.store, &repo_ids)
        .await
        .map_err(|e| e.to_string())?;
    Ok(trend.into_iter().map(TrendPointView::from).collect())
}

/// One board's row in the cross-board Portfolio: identity and daily trend series. No score or
/// tier; presented in board order, not ranked.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PortfolioBoardView {
    pub board_id: String,
    pub name: String,
    pub kind: String,
    pub trend: Vec<TrendPointView>,
}

/// The Portfolio overview: every board with its daily trend series. Boards are in stored order;
/// a descriptive overview, not a ranking. Read-only; works with no token.
#[tauri::command]
#[specta::specta]
async fn portfolio(state: State<'_, AppState>) -> Result<Vec<PortfolioBoardView>, String> {
    let boards = state.store.boards().await.map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(boards.len());
    for board in boards {
        let repo_ids = state
            .store
            .board_effective_repo_ids(&board.id)
            .await
            .map_err(|e| e.to_string())?;
        let trend = core_summary::board_trend(&state.store, &repo_ids)
            .await
            .map_err(|e| e.to_string())?;
        out.push(PortfolioBoardView {
            board_id: board.id,
            name: board.name,
            kind: board.kind.as_str().to_string(),
            trend: trend.into_iter().map(TrendPointView::from).collect(),
        });
    }
    Ok(out)
}

/// One metric's week-over-week movement for the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct MetricDeltaView {
    pub metric: String,
    pub current: i64,
    pub previous: i64,
    pub delta: i64,
    pub anomaly: bool,
    /// True when the anomaly was decided against the same-weekday baseline, so the UI can say
    /// "unusual for a <weekday>".
    pub seasonal: bool,
}

/// A board's "what changed" summary for the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct TrendDeltasView {
    pub days: i64,
    pub deltas: Vec<MetricDeltaView>,
}

impl From<core_summary::TrendDeltas> for TrendDeltasView {
    fn from(t: core_summary::TrendDeltas) -> Self {
        Self {
            days: t.days,
            deltas: t
                .deltas
                .into_iter()
                .map(|d| MetricDeltaView {
                    metric: d.metric,
                    current: d.current,
                    previous: d.previous,
                    delta: d.delta,
                    anomaly: d.anomaly,
                    seasonal: d.seasonal,
                })
                .collect(),
        }
    }
}

/// A board's week-over-week deltas and anomaly flags over its daily snapshots.
#[tauri::command]
#[specta::specta]
async fn board_trend_summary(
    state: State<'_, AppState>,
    id: String,
) -> Result<TrendDeltasView, String> {
    let repo_ids = state
        .store
        .board_effective_repo_ids(&id)
        .await
        .map_err(|e| e.to_string())?;
    let t = core_summary::trend_deltas(&state.store, &repo_ids)
        .await
        .map_err(|e| e.to_string())?;
    Ok(t.into())
}

/// Typed event the scheduler emits after polling one source: the source id and either the
/// per-type counts (`summary`, on success) or the `error` string.
/// A granular sync progress update. Boundary form of `core_sync::SyncProgress`. `phase` is
/// "planning" or "syncing"; `item` is a board (planning) or repo full-name (syncing) with its
/// position; `step` is the sub-step within a repo. When `finished`, the completion fields
/// (`source_id`, `ok`, `summary`, `error`) describe the repo's result.
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
pub struct SyncProgressEvent {
    pub phase: String,
    pub item: String,
    pub item_done: u32,
    pub item_total: u32,
    pub step: Option<String>,
    pub step_done: u32,
    pub step_total: u32,
    pub finished: bool,
    pub source_id: Option<String>,
    pub ok: bool,
    pub summary: Option<SyncSummary>,
    pub error: Option<String>,
}

/// A per-repo background-indexing update. Boundary form of `core_sync::IndexProgress`. `state`
/// is "indexing" | "indexed" | "error" (the host also derives "queued" from a fetch-completed
/// event). Drives the per-repo index badge.
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
pub struct IndexProgressEvent {
    pub repo_id: String,
    pub full_name: String,
    pub state: String,
    pub files: u32,
    /// Embedding sub-progress while `indexing` (units done / total this pass); 0 otherwise.
    /// Shown as a percentage on the index badge.
    pub done: u32,
    pub total: u32,
    /// The failure reason when `state` is "error" (else null).
    pub error: Option<String>,
    /// Files still to index when `state` is "partial"; 0 otherwise.
    pub pending: u32,
    /// Files this pass could not fetch when `state` is "partial"; 0 otherwise.
    pub skipped: u32,
    /// Why a "partial" or "paused" state holds. Separate from `error`: neither is a failure, and a
    /// red badge would send the user hunting for one.
    pub note: Option<String>,
}

impl From<core_sync::IndexProgress> for IndexProgressEvent {
    fn from(p: core_sync::IndexProgress) -> Self {
        Self {
            repo_id: p.repo_id,
            full_name: p.full_name,
            state: index_state_str(p.state).to_string(),
            files: p.files as u32,
            pending: p.pending as u32,
            skipped: p.skipped as u32,
            note: p.note,
            done: p.done as u32,
            total: p.total as u32,
            error: p.error,
        }
    }
}

impl From<SyncProgress> for SyncProgressEvent {
    fn from(p: SyncProgress) -> Self {
        let phase = match p.phase {
            SyncPhase::Planning => "planning",
            SyncPhase::Syncing => "syncing",
        }
        .to_string();
        let (finished, source_id, ok, summary, error) = match p.finished {
            Some(tick) => match tick.report {
                Ok(report) => (true, Some(tick.source_id), true, Some(report.into()), None),
                Err(e) => (true, Some(tick.source_id), false, None, Some(e)),
            },
            None => (false, None, false, None, None),
        };
        Self {
            phase,
            item: p.item,
            item_done: p.item_done as u32,
            item_total: p.item_total as u32,
            step: p.step,
            step_done: p.step_done as u32,
            step_total: p.step_total as u32,
            finished,
            source_id,
            ok,
            summary,
            error,
        }
    }
}

/// Build the tauri-specta `Builder` with the command and event surface. Shared by `run()` and
/// the bindings exporter test, so the registered surface and generated bindings match.
pub fn builder() -> Builder {
    Builder::<tauri::Wry>::new()
        .commands(collect_commands![
            health,
            open_url,
            sync_now,
            list_forges,
            add_forge,
            update_forge,
            delete_forge,
            set_forge_pat,
            disconnect_forge,
            connect_device_auth,
            github_oauth_available,
            attention,
            digest,
            pull_requests,
            change_digest,
            list_trackers,
            add_tracker,
            set_tracker_token,
            delete_tracker,
            link_board_tracker,
            unlink_board_tracker,
            board_jira,
            semantic_search,
            semantic_code_search,
            debug_stats,
            index_status,
            stored_repos,
            storage_report,
            storage_state,
            drop_repo_index,
            reclaim_free_space,
            forget_repo,
            orphan_repos,
            reclaim_orphan_repos,
            create_board,
            list_boards,
            delete_board,
            add_board_person,
            remove_board_person,
            add_board_repo,
            search_forge_users,
            search_forge_orgs,
            search_forge_repos,
            remove_board_repo,
            set_board_org,
            set_board_include_archived,
            set_repo_index_code,
            board_signals,
            set_board_signal_enabled,
            board_overview,
            board_dora,
            board_attention,
            board_changes,
            board_standup,
            person_activity,
            board_people_stats,
            board_pickup,
            board_investment,
            board_epics,
            start_board_brief,
            start_assistant_chat,
            llm_catalog,
            set_llm_settings,
            download_llm_model,
            portfolio,
            person_work_pattern,
            board_trend,
            board_trend_summary,
            board_hotspots,
            board_code_risk,
            board_code_health,
            board_ownership_risks,
            board_coupling,
            board_dependencies,
            board_scorecard,
            board_link_coverage,
            board_flow_metrics,
            board_cycle_phases,
            board_bug_flow,
            board_say_do,
            board_velocity_trend,
            board_flow_trend,
            board_digest,
            send_board_digest,
            set_board_scan_dependencies,
            scan_board_dependencies,
            get_settings,
            set_settings
        ])
        .events(collect_events![
            SyncProgressEvent,
            IndexProgressEvent,
            DeviceAuthEvent,
            BriefEvent,
            AssistantChatEvent,
            LlmDownloadEvent
        ])
}

#[cfg(test)]
mod tests {
    use super::{
        bound_board_attention, builder, resolve_github_client_id, AttentionView,
        BoardAttentionView, ATTENTION_SYSTEM, DEFAULT_GITHUB_CLIENT_ID,
    };
    use specta_typescript::{BigIntExportBehavior, Typescript};

    // ---- board_overview payload bound ----

    /// A minimal `AttentionView` of the given `kind`; only `kind` drives the severity sort.
    fn attention_view(kind: &str) -> AttentionView {
        AttentionView {
            kind: kind.into(),
            entities: Vec::new(),
            summary: format!("a {kind} item"),
            repo_full_name: None,
            ts: None,
            action: String::new(),
            wip_percentile: None,
            wip_percentile_basis: None,
        }
    }

    /// Under the cap, nothing is dropped and the reported total matches what is kept.
    #[test]
    fn bound_board_attention_is_a_no_op_under_the_cap() {
        let mut per_repo = vec![
            vec![attention_view("stale_pr"), attention_view("review_wait")],
            vec![attention_view("orphan_pr")],
        ];
        let total = bound_board_attention(&mut per_repo, 10);
        assert_eq!(total, 3);
        assert_eq!(per_repo[0].len() + per_repo[1].len(), 3);
    }

    /// Over the cap, exactly `cap` items survive, the most severe across every repo (not just the
    /// first), and the reported total is the true, unbounded count.
    #[test]
    fn bound_board_attention_keeps_the_most_severe_items_and_reports_the_true_total() {
        // failing_ci outranks stale_pr outranks orphan_pr (core_summary::ATTENTION_ORDER).
        let mut per_repo = vec![
            vec![
                attention_view("orphan_pr"),
                attention_view("orphan_pr"),
                attention_view("failing_ci"),
            ],
            vec![attention_view("stale_pr"), attention_view("orphan_pr")],
        ];
        let total = bound_board_attention(&mut per_repo, 2);
        assert_eq!(total, 5, "the true total, not the bounded count");
        let kept: Vec<&str> = per_repo.iter().flatten().map(|a| a.kind.as_str()).collect();
        assert_eq!(kept.len(), 2, "exactly the cap survives");
        assert!(
            kept.contains(&"failing_ci"),
            "the most severe item must survive: {kept:?}"
        );
        assert!(
            kept.contains(&"stale_pr"),
            "the second most severe item must survive: {kept:?}"
        );
        assert!(
            !kept.contains(&"orphan_pr"),
            "the least severe kind is what gets cut: {kept:?}"
        );
    }

    // ---- typed attention envelope across the boundary ----

    fn envelope_item(kind: &str) -> core_summary::AttentionItem {
        core_summary::AttentionItem {
            kind: kind.into(),
            entities: vec![
                core_model::LinkedEntity::subject(core_model::AttentionEntity::PullRequest {
                    id: "p1".into(),
                    number: 12,
                    title: "Fix the thing".into(),
                    url: Some("https://forge/acme/widget/pull/12".into()),
                }),
                core_model::LinkedEntity::actor(core_model::AttentionEntity::Person {
                    login: "octocat".into(),
                }),
                core_model::LinkedEntity::evidence(core_model::AttentionEntity::WorkItem {
                    id: "i1".into(),
                    number: 42,
                    title: "a bug".into(),
                    url: Some("https://forge/acme/widget/issues/42".into()),
                }),
            ],
            summary: "PR #12 merged but #42 is still open".into(),
            ts: Some("2026-06-03T00:00:00Z".into()),
            action: "Reopen the issue, or confirm the PR closes it.".into(),
            wip_percentile: None,
            wip_percentile_basis: None,
        }
    }

    /// The whole typed envelope crosses the boundary intact, in particular the `evidence` entity.
    #[test]
    fn the_attention_view_carries_the_whole_envelope_across_the_boundary() {
        let item = envelope_item("done_not_done");
        let view = AttentionView::from(item.clone());
        assert_eq!(
            view.entities, item.entities,
            "no entity is dropped or reordered"
        );
        assert_eq!(view.summary, item.summary);
        assert_eq!(view.action, item.action);
        // The repo is the one thing the host adds; the core does not know it.
        assert_eq!(view.repo_full_name, None);

        let board = core_summary::BoardAttentionItem {
            repo_id: "repo:acme/widget".into(),
            kind: item.kind.clone(),
            entities: item.entities.clone(),
            summary: item.summary.clone(),
            by_team: true,
            ts: item.ts.clone(),
            wip_percentile: None,
            wip_percentile_basis: None,
        };
        let board_view = BoardAttentionView::from(board);
        assert_eq!(board_view.entities, item.entities);
        assert!(board_view.by_team);
    }

    /// The narrator's system prompt must say explicitly not to produce a list or one line per item.
    #[test]
    fn attention_system_prompt_forbids_a_list() {
        let lower = ATTENTION_SYSTEM.to_lowercase();
        assert!(
            lower.contains("not write a list") || lower.contains("not just list"),
            "prompt must tell the model not to emit a list: {ATTENTION_SYSTEM}"
        );
        assert!(
            lower.contains("one line per item"),
            "prompt must rule out one-line-per-item output: {ATTENTION_SYSTEM}"
        );
    }

    /// The UI narrows the entity union on its `type` discriminant, which `tauri-specta` generates.
    /// A dropped `Type` derive or a changed serde tag would break narrowing, so pin it here rather
    /// than only in `just ui-check`.
    #[test]
    fn the_exported_bindings_carry_the_entity_union() {
        let dir = std::env::temp_dir().join("orgonzola-bindings-spec119");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bindings.ts");
        builder()
            .export(
                Typescript::default().bigint(BigIntExportBehavior::Number),
                &path,
            )
            .expect("failed to export typescript bindings");
        let ts = std::fs::read_to_string(&path).unwrap();
        for discriminant in [
            r#"type: "pull_request""#,
            r#"type: "work_item""#,
            r#"type: "ci_run""#,
            r#"type: "person""#,
            r#"type: "source_file""#,
        ] {
            assert!(
                ts.contains(discriminant),
                "the generated AttentionEntity union must carry {discriminant}"
            );
        }
        assert!(ts.contains(
            "export type LinkedEntity = { role: AttentionRole; target: AttentionEntity }"
        ));
        assert!(ts.contains(r#"export type AttentionRole = "#));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn github_client_id_prefers_the_forge_then_the_default() {
        // A forge's own client id wins.
        assert_eq!(
            resolve_github_client_id(Some("forge-cid")).as_deref(),
            Some("forge-cid")
        );
        // With no forge client id, fall back to the build-time default (empty in this build ->
        // None).
        let fallback = resolve_github_client_id(None);
        if DEFAULT_GITHUB_CLIENT_ID.is_empty() {
            assert_eq!(fallback, None);
        } else {
            assert_eq!(fallback.as_deref(), Some(DEFAULT_GITHUB_CLIENT_ID));
        }
        // An empty per-forge id is treated as unset.
        assert_eq!(resolve_github_client_id(Some("")), fallback);
    }

    // ---- storage accounting ----

    /// The walk is checked against files of known size: nested directories are included, and the
    /// count says how many files were seen.
    #[test]
    fn dir_bytes_sums_nested_files() {
        let root = std::env::temp_dir().join(format!("orgonzola-dirsize-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("nested/deeper")).unwrap();
        std::fs::write(root.join("a.onnx"), vec![b'x'; 30]).unwrap();
        std::fs::write(root.join("nested/b.bin"), vec![b'y'; 100]).unwrap();
        std::fs::write(root.join("nested/deeper/c.txt"), vec![b'z'; 7]).unwrap();

        assert_eq!(super::dir_bytes(&root), (137, 3));
        // A single file answers for itself, so an asset that is one GGUF is still sized.
        assert_eq!(super::dir_bytes(&root.join("a.onnx")), (30, 1));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Bundled model dirs are absent in a dev build without `fastembed`, and an unconfigured path
    /// is normal too. Neither may fail the report or invent a size.
    #[test]
    fn a_missing_asset_reports_absent_rather_than_failing() {
        let missing = std::env::temp_dir().join("orgonzola-no-such-model-dir");
        let _ = std::fs::remove_dir_all(&missing);
        let a = super::asset("Embedding model", "embed_model", Some(&missing), false);
        assert!(!a.exists);
        assert_eq!((a.bytes, a.files), (0, 0));
        assert_eq!(a.path, missing.to_string_lossy());

        let unset = super::asset("Bundled LLM", "llm_bundled", None, false);
        assert!(!unset.exists);
        assert_eq!(unset.bytes, 0);
        assert_eq!(unset.path, "not configured");
    }

    /// The report must reach the UI: a dropped `Type` derive or a command missing from
    /// `collect_commands!` would generate bindings the Debug view cannot call.
    #[test]
    fn the_exported_bindings_carry_the_storage_report() {
        let dir = std::env::temp_dir().join("orgonzola-bindings-spec120");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bindings.ts");
        builder()
            .export(
                Typescript::default().bigint(BigIntExportBehavior::Number),
                &path,
            )
            .expect("failed to export typescript bindings");
        let ts = std::fs::read_to_string(&path).unwrap();
        for needle in [
            "storageReport",
            "StorageReportView",
            "RepoStorageView",
            "AssetStorageView",
            "StorageGroupView",
        ] {
            assert!(ts.contains(needle), "the bindings are missing {needle}");
        }
        // Byte counts must not cross as `bigint`: the UI formats and sums them as numbers.
        assert!(
            !ts.contains("reserved_bytes: bigint"),
            "byte figures crossed the boundary as bigint"
        );
    }

    // ---- storage budget ----

    /// Every index-lane state needs a wire string the UI knows. A string the UI does not know would
    /// render no badge, making a stopped repo look up to date.
    #[test]
    fn every_index_state_has_a_wire_string_the_ui_knows() {
        use core_sync::IndexState::*;
        for (state, wire) in [
            (Queued, "queued"),
            (Indexing, "indexing"),
            (Indexed, "indexed"),
            (Partial, "partial"),
            (Paused, "paused"),
            (Error, "error"),
        ] {
            assert_eq!(super::index_state_str(state), wire);
        }
        // The UI's badge map is keyed by these strings; a duplicate would make two states render
        // alike.
        let all = [Queued, Indexing, Indexed, Partial, Paused, Error]
            .map(super::index_state_str)
            .to_vec();
        let mut unique = all.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            all.len(),
            unique.len(),
            "two index states share a wire name"
        );
    }

    /// The budget is set in one view and enforced in another, both through the generated bindings.
    /// A missing command or `Type` derive would surface as an unclear TypeScript error in
    /// `just ui-check`.
    #[test]
    fn the_exported_bindings_carry_the_storage_budget() {
        let dir = std::env::temp_dir().join("orgonzola-bindings-spec121");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bindings.ts");
        builder()
            .export(
                Typescript::default().bigint(BigIntExportBehavior::Number),
                &path,
            )
            .expect("failed to export typescript bindings");
        let ts = std::fs::read_to_string(&path).unwrap();
        for needle in [
            "storageState",
            "StorageStateView",
            "dropRepoIndex",
            "DroppedIndexView",
            "reclaimFreeSpace",
            "ReclaimReportView",
            "storageBudgetMb",
        ] {
            assert!(ts.contains(needle), "the bindings are missing {needle}");
        }
    }

    /// The bulk orphan reclaim in Settings > Storage reaches the core through the generated
    /// bindings.
    #[test]
    fn the_exported_bindings_carry_the_orphan_reclaim() {
        let dir = std::env::temp_dir().join("orgonzola-bindings-spec134");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bindings.ts");
        builder()
            .export(
                Typescript::default().bigint(BigIntExportBehavior::Number),
                &path,
            )
            .expect("failed to export typescript bindings");
        let ts = std::fs::read_to_string(&path).unwrap();
        for needle in [
            "orphanRepos",
            "OrphanRepoReportView",
            "reclaimOrphanRepos",
            "OrphanReclaimView",
        ] {
            assert!(ts.contains(needle), "the bindings are missing {needle}");
        }
    }

    /// Regenerate the UI's typed bindings from the Rust surface. Runs headless
    /// (`cargo test -p orgonzola-desktop --lib`); a Rust signature change shows up as a
    /// TypeScript diff.
    #[test]
    fn export_typescript_bindings() {
        // Map Rust integers wider than JS's safe range (u64 ids/timestamps) to `number`. GitHub ids
        // stay within 2^53; revisit if a field needs the full u64 range.
        let path = "../../ui/src/bindings.ts";
        builder()
            .export(
                Typescript::default().bigint(BigIntExportBehavior::Number),
                path,
            )
            .expect("failed to export typescript bindings");

        // specta emits trailing whitespace on some lines and `ui/biome.json` ignores this file, so
        // strip trailing whitespace per line and keep a single trailing newline. That keeps the
        // committed
        // file identical to the test output.
        let generated = std::fs::read_to_string(path).expect("failed to read generated bindings");
        let mut normalized: String = generated
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        normalized.push('\n');
        if normalized != generated {
            std::fs::write(path, normalized).expect("failed to write normalized bindings");
        }
    }
}

/// Diagnostic: run the real search path (real model + real DB) against the live index. Ignored
/// by default; run with the live DB to see what semantic search returns:
///   ORG_DB=~/.local/share/com.kikijiki.orgonzola/orgonzola.db \
///   ORG_Q="how does sync route a repo" \
///   cargo test -p orgonzola-desktop --features fastembed --lib diag_live_search \
///     -- --ignored --nocapture
#[cfg(all(test, feature = "fastembed"))]
mod diag_search {
    use super::*;
    use core_store::{KindFilter, Store};

    #[tokio::test]
    #[ignore = "needs the live DB + bundled model; diagnostic"]
    async fn diag_live_search() {
        let src = std::env::var("ORG_DB").unwrap_or_else(|_| {
            format!(
                "{}/.local/share/com.kikijiki.orgonzola/orgonzola.db",
                std::env::var("HOME").unwrap()
            )
        });
        let query = std::env::var("ORG_Q").unwrap_or_else(|_| "authentication token flow".into());
        let code = std::env::var("ORG_CODE").is_ok();

        // Copy the live DB (+wal/shm) to a temp dir so the app's file and lock are untouched.
        let tmp = std::env::temp_dir().join("orgonzola-diag");
        let _ = std::fs::create_dir_all(&tmp);
        let dst = tmp.join("orgonzola.db");
        for suf in ["", "-wal", "-shm"] {
            let s = format!("{src}{suf}");
            if std::path::Path::new(&s).exists() {
                std::fs::copy(&s, format!("{}{suf}", dst.display())).unwrap();
            }
        }

        let store = Store::open(&dst.to_string_lossy()).await.unwrap();
        let models = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/models");
        let embedder = build_embedder(Some(models.join("jina-embeddings-v2-base-code")));
        let reranker = build_reranker(Some(models.join("jina-reranker-v1-turbo-en")));
        eprintln!(
            "embedder={} dims={} reranker={}",
            embedder.name(),
            embedder.dimensions(),
            reranker.name()
        );

        let filter = if code {
            KindFilter::Is("code")
        } else {
            KindFilter::Not("code")
        };
        let qvec = embedder.embed(&query);
        eprintln!(
            "\nQUERY: {query:?}  (kind={})  |qvec|={:.4}",
            if code { "code" } else { "activity" },
            qvec.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Vector arm alone. Unscoped on purpose: a retrieval-quality probe over the whole index.
        let vec = store
            .vector_search(&qvec, 10, filter, core_store::RepoScope::All)
            .await
            .unwrap();
        eprintln!("\n-- vector arm (cosine) --");
        for h in &vec {
            eprintln!("  {:.3}  {}  {}", h.score, h.ref_id, oneline(&h.chunk));
        }

        // Fused hybrid.
        let hits = store
            .hybrid_search(&qvec, &query, 10, filter, core_store::RepoScope::All)
            .await
            .unwrap();
        eprintln!("\n-- hybrid fused (RRF) --");
        for h in &hits {
            eprintln!("  {:.4}  {}  {}", h.score, h.ref_id, oneline(&h.chunk));
        }

        // After rerank.
        let docs: Vec<String> = hits.iter().map(|h| h.chunk.clone()).collect();
        let order = reranker.rerank(&query, &docs);
        eprintln!("\n-- after cross-encoder rerank --");
        for i in order.into_iter().take(10) {
            if let Some(h) = hits.get(i) {
                eprintln!("  {}  {}", h.ref_id, oneline(&h.chunk));
            }
        }
    }

    fn oneline(s: &str) -> String {
        let t: String = s.chars().take(70).collect();
        t.replace('\n', " ")
    }
}

/// Credential-aware host logic over the fake `CredentialStore`, headless. The device-flow
/// handshake is verified on the desktop.
#[cfg(test)]
mod cred_tests {
    use super::*;
    use core_store::{ForgeRow, Store};

    async fn store_with_github_forge(id: &str) -> Store {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_forge(&ForgeRow {
                id: id.into(),
                name: "GitHub".into(),
                kind: "github".into(),
                base_url: "https://api.github.com".into(),
                oauth_client_id: None,
            })
            .await
            .unwrap();
        store
    }

    #[tokio::test]
    async fn build_engine_skips_forges_without_a_credential() {
        let store = store_with_github_forge("gh").await;
        let embedder: Arc<dyn core_embed::Embedder> = Arc::new(core_embed::HashEmbedder::default());
        let creds = MemoryCredentials::default();
        // No credential in the keychain -> nothing to build -> no engine.
        assert!(build_engine(store.clone(), embedder.clone(), &creds)
            .await
            .is_none());
        // Store a token -> the forge is built and the engine exists.
        creds.set("gh", "tok").unwrap();
        assert!(build_engine(store, embedder, &creds).await.is_some());
    }

    // ---- board Search tab scope ----

    /// The Search tab resolves the board's effective repo set: pinned if any, otherwise discovered,
    /// the same rule as the rest of the board and the assistant.
    #[tokio::test]
    async fn board_search_scope_resolves_the_boards_repos_and_refuses_an_unknown_board() {
        use core_store::BoardKind;
        let store = Store::open_in_memory().await.unwrap();
        store
            .create_board(
                "board:platform",
                "Platform",
                BoardKind::Team,
                "2026-08-01T00:00:00Z",
            )
            .await
            .unwrap();

        // No repos yet -> an empty scope. `RepoScope::Repos(&[])` matches nothing; it must never
        // widen to the whole store.
        assert!(board_search_scope(&store, "board:platform")
            .await
            .unwrap()
            .is_empty());

        // Discovered repos are the scope while nothing is pinned.
        store
            .replace_board_discovered_repos("board:platform", &["repo:gh/acme/found".to_string()])
            .await
            .unwrap();
        assert_eq!(
            board_search_scope(&store, "board:platform").await.unwrap(),
            vec!["repo:gh/acme/found"]
        );

        // Pinning takes over, matching `board_effective_repo_ids`.
        store
            .add_board_repo("board:platform", "repo:gh/acme/pinned")
            .await
            .unwrap();
        assert_eq!(
            board_search_scope(&store, "board:platform").await.unwrap(),
            vec!["repo:gh/acme/pinned"]
        );

        // An unknown board is an error naming it, not an unscoped search.
        let err = board_search_scope(&store, "board:ghost").await.unwrap_err();
        assert!(
            err.contains("board:ghost"),
            "the error must name the board: {err}"
        );
    }

    #[tokio::test]
    async fn forge_view_reflects_connection_and_omits_the_token() {
        let creds = MemoryCredentials::default();
        let row = ForgeRow {
            id: "gh".into(),
            name: "GitHub".into(),
            kind: "github".into(),
            base_url: "https://api.github.com".into(),
            oauth_client_id: Some("cid".into()),
        };
        assert!(!forge_view(row.clone(), &creds).connected);
        creds.set("gh", "super-secret-token").unwrap();
        let v = forge_view(row, &creds);
        assert!(v.connected);
        // The token is not a field of ForgeView, so even a full serialization cannot leak it.
        let json = serde_json::to_string(&v).unwrap();
        assert!(
            !json.contains("super-secret-token"),
            "ForgeView must not carry the token"
        );
    }

    // A connection names itself: github.com by brand, anything else by its host.
    #[test]
    fn a_blank_connection_name_is_derived_from_the_kind_and_host() {
        assert_eq!(
            default_connection_name("github", "https://api.github.com"),
            "GitHub"
        );
        assert_eq!(default_connection_name("github", ""), "GitHub");
        assert_eq!(
            default_connection_name("github", "https://ghe.acme.com/api/v3"),
            "ghe.acme.com"
        );
        assert_eq!(
            default_connection_name("gitea", "https://git.example.com:3000/api/v1"),
            "git.example.com"
        );
        assert_eq!(
            default_connection_name("jira", "https://acme.atlassian.net"),
            "acme.atlassian.net"
        );
    }

    #[test]
    fn a_derived_name_steps_around_the_ones_already_taken() {
        let taken = ["GitHub".to_string(), "GitHub (2)".to_string()];
        assert_eq!(
            unique_name("GitHub", taken.iter().map(String::as_str)),
            "GitHub (3)"
        );
        assert_eq!(unique_name("GitHub", std::iter::empty()), "GitHub");
    }

    #[test]
    fn url_host_strips_the_scheme_userinfo_port_and_path() {
        assert_eq!(
            url_host("https://Git.Example.com:3000/api/v1").as_deref(),
            Some("git.example.com")
        );
        assert_eq!(
            url_host("http://user@git.example.com/api").as_deref(),
            Some("git.example.com")
        );
        assert_eq!(url_host("git.example.com"), None);
        assert_eq!(url_host(""), None);
    }

    #[test]
    fn a_blank_url_defaults_for_github_and_is_an_error_for_the_rest() {
        let (name, url) = resolve_connection_fields("", "github", "", &[]).unwrap();
        assert_eq!(
            (name.as_str(), url.as_str()),
            ("GitHub", "https://api.github.com")
        );

        // A missing URL must fail, naming what is missing, rather than silently point a
        // connection at an example host.
        for kind in ["gitea", "jira"] {
            let err = resolve_connection_fields("", kind, "", &[]).unwrap_err();
            assert!(
                err.contains("URL"),
                "{kind}: expected a missing-URL error, got {err}"
            );
        }
    }

    // Connecting takes effect in the running process, without a restart.
    #[tokio::test]
    async fn the_engine_follows_the_credential_within_one_process() {
        let store = store_with_github_forge("gh").await;
        let embedder: Arc<dyn core_embed::Embedder> = Arc::new(core_embed::HashEmbedder::default());
        let creds = MemoryCredentials::default();
        let state = AppState::new(
            store.clone(),
            embedder.clone(),
            Arc::new(core_embed::NoopReranker),
            Arc::new(MemoryCredentials::default()),
            "db".into(),
            std::path::PathBuf::from("."),
            None,
        );
        assert!(state.engine().is_none(), "nothing authorized yet");

        // Connect: the engine appears without the process restarting.
        creds.set("gh", "tok").unwrap();
        let built = build_engine(store.clone(), embedder.clone(), &creds).await;
        assert!(state.swap_engine(built.map(Arc::new), Vec::new()));
        assert!(
            state.engine().is_some(),
            "a stored credential means a live engine"
        );

        // Disconnect: it goes away again.
        creds.delete("gh").unwrap();
        let built = build_engine(store, embedder, &creds).await;
        assert!(!state.swap_engine(built.map(Arc::new), Vec::new()));
        assert!(state.engine().is_none());
    }

    // The previous engine's tasks must stop when it is replaced, or an edited or disconnected
    // connection keeps syncing under the old forge set.
    #[tokio::test]
    async fn a_swap_aborts_the_tasks_of_the_engine_it_replaces() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // A guard whose Drop fires when the task's future is dropped, as an abort does.
        struct SetOnDrop(Arc<AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let state = AppState::new(
            store_with_github_forge("gh").await,
            Arc::new(core_embed::HashEmbedder::default()),
            Arc::new(core_embed::NoopReranker),
            Arc::new(MemoryCredentials::default()),
            "db".into(),
            std::path::PathBuf::from("."),
            None,
        );

        let cancelled = Arc::new(AtomicBool::new(false));
        let guard = SetOnDrop(cancelled.clone());
        let task = tauri::async_runtime::spawn(async move {
            let _guard = guard;
            // Never finishes on its own, so only an abort can end it (like the scheduler loop).
            std::future::pending::<()>().await;
        });
        state.swap_engine(None, vec![task]);
        state.swap_engine(None, Vec::new());

        // The abort is asynchronous; poll for the drop rather than sleeping a fixed time.
        for _ in 0..100 {
            if cancelled.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            cancelled.load(Ordering::SeqCst),
            "swapping the engine must abort the task the previous one owned"
        );
    }

    // "Could not search" and "searched, found nothing" are different answers.
    #[tokio::test]
    async fn an_unrunnable_search_is_an_error_not_an_empty_result() {
        let store = store_with_github_forge("gh").await;
        let embedder: Arc<dyn core_embed::Embedder> = Arc::new(core_embed::HashEmbedder::default());
        let creds = MemoryCredentials::default();
        let state = AppState::new(
            store.clone(),
            embedder.clone(),
            Arc::new(core_embed::NoopReranker),
            Arc::new(MemoryCredentials::default()),
            "db".into(),
            std::path::PathBuf::from("."),
            None,
        );

        // Nothing to search for is the one case that is legitimately "no answer".
        assert!(matches!(search_context(&state, "gh", "   "), Ok(None)));

        // Nothing authorized: an error naming the step.
        let err = search_context(&state, "gh", "kikijiki")
            .err()
            .expect("no connection authorized -> an error");
        assert!(err.contains("Settings"), "{err}");

        // A board with no connection assigned names that instead.
        let err = search_context(&state, "", "kikijiki")
            .err()
            .expect("no board connection -> an error");
        assert!(err.contains("board"), "{err}");

        // With a credential the search is runnable.
        creds.set("gh", "tok").unwrap();
        let built = build_engine(store, embedder, &creds).await;
        state.swap_engine(built.map(Arc::new), Vec::new());
        assert!(matches!(
            search_context(&state, "gh", "kikijiki"),
            Ok(Some(_))
        ));
    }

    #[test]
    fn the_no_connection_message_points_at_settings_not_at_a_dotfile() {
        // The message must not name GITHUB_TOKEN or .env.local; that is not how the app is
        // configured.
        assert!(!NO_CONNECTION.contains("GITHUB_TOKEN"));
        assert!(!NO_CONNECTION.contains(".env"));
        assert!(NO_CONNECTION.contains("Settings"));
    }

    #[test]
    fn an_explicit_name_and_url_are_left_alone() {
        let (name, url) =
            resolve_connection_fields("Work GitHub", "github", "https://ghe.acme.com", &[])
                .unwrap();
        assert_eq!(
            (name.as_str(), url.as_str()),
            ("Work GitHub", "https://ghe.acme.com")
        );
    }
}
