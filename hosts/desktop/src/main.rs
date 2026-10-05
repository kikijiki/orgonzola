//! The desktop shell binary. Spawns the core ticker on a Tokio task and forwards each tick to
//! the UI as a typed event.

// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use std::time::Duration;

use orgonzola_desktop::{
    build_embedder, build_reranker, build_store, builder, install_engine, AppState,
    CredentialStore, KeyringCredentials,
};
use tauri::path::BaseDirectory;
use tauri::Manager;

/// Load `.env.local` then `.env`, searching upward from the current directory.
/// `.env.local` wins because dotenvy does not overwrite a variable already set.
fn load_env() {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    for name in [".env.local", ".env"] {
        if let Some(path) = cwd.ancestors().map(|d| d.join(name)).find(|p| p.is_file()) {
            let _ = dotenvy::from_path(&path);
        }
    }
}

fn main() {
    load_env();

    let builder = builder();

    tauri::Builder::default()
        .invoke_handler(builder.invoke_handler())
        .setup(move |app| {
            builder.mount_events(app);

            // Open the store, then build the sync engine over a shared clone. The engine is
            // None without GITHUB_TOKEN; source management and health still work.
            let db_path = app
                .path()
                .app_data_dir()
                .map(|dir| {
                    let _ = std::fs::create_dir_all(&dir);
                    dir.join("orgonzola.db")
                })
                .unwrap_or_else(|_| std::path::PathBuf::from("orgonzola.db"));
            let store = tauri::async_runtime::block_on(build_store(&db_path.to_string_lossy()))
                .expect("failed to open the orgonzola store");
            // Bundled model directories. Without the `fastembed` feature the builders fall back to
            // the deterministic embedder and a no-op reranker.
            let resolve = |sub: &str| app.path().resolve(sub, BaseDirectory::Resource).ok();
            // Kept because the storage report sizes these directories.
            let embed_model_dir = resolve("models/jina-embeddings-v2-base-code");
            let rerank_model_dir = resolve("models/jina-reranker-v1-turbo-en");
            let embedder = build_embedder(embed_model_dir.clone());
            let reranker = build_reranker(rerank_model_dir.clone());
            // Downloaded LLM models go to the OS app-data dir, next to the DB. The bundled default
            // model comes from the env dir `just run` sets in dev, else the app resource dir.
            let llm_download_dir = app
                .path()
                .app_data_dir()
                .map(|dir| dir.join("models/llm"))
                .unwrap_or_else(|_| std::path::PathBuf::from("models/llm"));
            let _ = std::fs::create_dir_all(&llm_download_dir);
            let llm_bundled_dir = std::env::var("ORGONZOLA_LLM_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .map(std::path::PathBuf::from)
                .or_else(|| resolve("models/llm"));
            // Forge access tokens live in the OS keychain, not the DB.
            let credentials: Arc<dyn CredentialStore> = Arc::new(KeyringCredentials::new());
            app.manage(
                AppState::new(
                    store.clone(),
                    embedder,
                    reranker,
                    credentials,
                    db_path.to_string_lossy().into_owned(),
                    llm_download_dir,
                    llm_bundled_dir,
                )
                .with_model_dirs(embed_model_dir, rerank_model_dir),
            );

            // Install the sync engine plus the index worker and scheduler that own it. The
            // connect/disconnect/edit commands call `install_engine` again when connections
            // change. With nothing authorized it installs nothing and the shell stays read-only.
            let handle = app.handle().clone();
            tauri::async_runtime::block_on(install_engine(&handle));

            // Record each board's flow and bug metrics for today (idempotent per day), at startup
            // and every 6 hours.
            {
                let store = store.clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        orgonzola_desktop::record_board_snapshots(&store).await;
                        tokio::time::sleep(Duration::from_secs(6 * 60 * 60)).await;
                    }
                });
            }

            // POST each board's digest every `digest_schedule_hours` when a cadence and webhook
            // are set. The last-send time is persisted in `meta` (epoch seconds) across restarts.
            {
                let store = store.clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        tokio::time::sleep(Duration::from_secs(30 * 60)).await;
                        let Ok(settings) = store.settings().await else {
                            continue;
                        };
                        let Some(hours) = settings.digest_schedule_hours.filter(|h| *h > 0) else {
                            continue;
                        };
                        if settings
                            .digest_webhook_url
                            .as_deref()
                            .map(str::trim)
                            .unwrap_or("")
                            .is_empty()
                        {
                            continue; // no webhook -> nothing to send (egress stays opt-in)
                        }
                        let now_epoch = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);
                        let due = match store.meta_get("digest_last_sent_epoch").await {
                            Ok(Some(last)) => last
                                .parse::<i64>()
                                .map(|l| now_epoch - l >= hours * 3600)
                                .unwrap_or(true),
                            _ => true, // never sent -> due now
                        };
                        if !due {
                            continue;
                        }
                        let since_days = ((hours + 23) / 24).max(1);
                        let Ok(boards) = store.boards().await else {
                            continue;
                        };
                        for board in boards {
                            if let Err(e) = orgonzola_desktop::push_board_digest(
                                &store, &board.id, since_days, 0,
                            )
                            .await
                            {
                                eprintln!("scheduled digest for board {} failed: {e}", board.id);
                            }
                        }
                        let _ = store
                            .meta_set("digest_last_sent_epoch", &now_epoch.to_string())
                            .await;
                    }
                });
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the orgonzola desktop shell");
}
