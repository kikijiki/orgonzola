//! Drive a real sync against a GitHub repo, end to end.
//! Usage (inside the devshell, with a token available):
//!   set -a; . ./.env.local; set +a
//!   cargo run -p core-sync --example sync -- kikijiki/orgonzola
//! Loads `.env.local`/`.env` for `GITHUB_TOKEN`, syncs into an in-memory store, prints results.

use core_forge_github::GithubForge;
use core_github::{EnvTokenProvider, GithubClient, HttpTransport};
use core_store::{Repo, Store};
use core_sync::SyncEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(".env.local");
    let _ = dotenvy::dotenv();

    let full_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "kikijiki/orgonzola".to_string());
    let (owner, name) = full_name
        .split_once('/')
        .ok_or("argument must be owner/repo")?;

    let token = std::env::var("GITHUB_TOKEN").map_err(|_| "GITHUB_TOKEN is not set")?;

    let store = Store::open_in_memory().await?;
    let transport = HttpTransport::new("https://api.github.com", token)?;
    let forge = GithubForge::new(GithubClient::new(transport, EnvTokenProvider::github()));
    let engine = SyncEngine::new(forge, store);

    // The forge is registered under `DEFAULT_FORGE`; ids are `repo:<forge>/<owner>/<name>`.
    let repo = Repo {
        id: core_sync::repo_id(core_sync::DEFAULT_FORGE, &full_name),
        owner: owner.to_string(),
        name: name.to_string(),
        full_name: full_name.clone(),
        ownership: "owned".to_string(),
    };

    println!("syncing {full_name} ...");
    let commits = engine.sync_repo(&repo).await?;
    let pulls = engine.sync_pull_requests(&repo).await?;

    println!("commits: {commits:?}");
    println!("pull requests: {pulls:?}");
    println!(
        "stored: {} commits, {} pull requests; rate-limit remaining seen on responses",
        engine.store().count_commits().await?,
        engine.store().pull_requests(&repo.id).await?.len(),
    );
    Ok(())
}
