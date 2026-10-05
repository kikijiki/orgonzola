-- orgonzola schema. Timestamps are ISO-8601 TEXT. Ids are TEXT: forge ids, or composite ids such as
-- "repo:<forge_id>/<owner>/<name>".
--
-- Two kinds of table. User-authored config (forges, trackers, sources, settings, boards and what hangs
-- off them) is the source of truth. Everything else is a cache that can be re-fetched or recomputed.
-- See the storage ADR ("rebuildable cache with a hard ceiling").

-- ---- connections and settings ------------------------------------------------

-- A configured code host. Holds no secret: the token lives in the OS keychain, keyed by `id` (see the
-- credentials ADR). `oauth_client_id` is public.
CREATE TABLE forges (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL,    -- 'github' | 'gitea'
    base_url        TEXT NOT NULL,
    oauth_client_id TEXT
);

-- A configured issue tracker. Separate from forges (see the Jira ADR).
CREATE TABLE trackers (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    kind     TEXT NOT NULL,   -- 'jira'
    base_url TEXT NOT NULL,
    email    TEXT             -- Jira Cloud basic-auth email; NULL for a bearer PAT (Server/DC)
);

-- Watch registry the scheduler polls. `core_model::Source` is the in-memory form.
CREATE TABLE sources (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,    -- 'org' | 'repo' | 'user'
    name          TEXT NOT NULL,    -- 'owner/name' for a repo, the login for an org or user
    ownership     TEXT NOT NULL,    -- 'owned' | 'observed'
    filters       TEXT NOT NULL,    -- JSON array of include filters
    stale_pr_days INTEGER NOT NULL,
    forge_id      TEXT REFERENCES forges (id)
);

-- Single row (id pinned to 1), seeded below so reads always find it.
CREATE TABLE settings (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    sync_period_secs      INTEGER NOT NULL,
    stale_pr_days         INTEGER NOT NULL DEFAULT 7,
    digest_webhook_url    TEXT,                         -- NULL = no egress
    digest_schedule_hours INTEGER,                      -- NULL = scheduled digest off
    llm_enabled           INTEGER NOT NULL DEFAULT 1,
    llm_model             TEXT,                         -- GGUF filename; NULL = env default
    storage_budget_mb     INTEGER DEFAULT 5120          -- NULL = no limit
);

INSERT INTO settings (id, sync_period_secs) VALUES (1, 300);

-- Scalar key/value state, e.g. the embedder id the index was built with.
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---- boards -----------------------------------------------------------------
-- A board is the unit of scope. Its kind is fixed at creation (see the board-kinds ADR).

CREATE TABLE boards (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    org              TEXT,                                 -- narrows discovery to one org
    forge_id         TEXT REFERENCES forges (id),
    scan_dependencies INTEGER NOT NULL DEFAULT 0,          -- opt-in OSV lookups (network egress)
    kind             TEXT NOT NULL DEFAULT 'team',         -- 'team' | 'repo' | 'org'
    include_archived INTEGER NOT NULL DEFAULT 0 CHECK (include_archived IN (0, 1))
);

-- People are forge logins, not FK'd to identities: a board can name someone before any of their
-- activity is synced.
CREATE TABLE board_people (
    board_id TEXT NOT NULL REFERENCES boards (id) ON DELETE CASCADE,
    login    TEXT NOT NULL,
    PRIMARY KEY (board_id, login)
);

-- Pinned repos. They override discovery.
CREATE TABLE board_repos (
    board_id TEXT NOT NULL REFERENCES boards (id) ON DELETE CASCADE,
    repo_id  TEXT NOT NULL,
    PRIMARY KEY (board_id, repo_id)
);

-- Repos found from the board's people. Written by sync, read without network.
CREATE TABLE board_discovered_repos (
    board_id TEXT NOT NULL REFERENCES boards (id) ON DELETE CASCADE,
    repo_id  TEXT NOT NULL,
    PRIMARY KEY (board_id, repo_id)
);

CREATE TABLE board_disabled_signals (
    board_id TEXT NOT NULL REFERENCES boards (id) ON DELETE CASCADE,
    signal   TEXT NOT NULL,
    PRIMARY KEY (board_id, signal)
);

CREATE TABLE board_trackers (
    board_id    TEXT NOT NULL REFERENCES boards (id),
    tracker_id  TEXT NOT NULL REFERENCES trackers (id),
    project_key TEXT NOT NULL,
    PRIMARY KEY (board_id, tracker_id, project_key)
);

-- Per-repo code indexing toggle. Absent row = on.
CREATE TABLE repo_index_prefs (
    repo_id    TEXT PRIMARY KEY,
    index_code INTEGER NOT NULL DEFAULT 1
);

-- ---- forge activity ---------------------------------------------------------

CREATE TABLE repos (
    id             TEXT PRIMARY KEY,
    owner          TEXT NOT NULL,
    name           TEXT NOT NULL,
    full_name      TEXT NOT NULL,      -- not unique: the same owner/name can exist on two forges
    ownership      TEXT NOT NULL CHECK (ownership IN ('owned', 'observed')),
    is_fork        INTEGER NOT NULL DEFAULT 0,
    parent_repo_id TEXT
);

CREATE TABLE commits (
    sha          TEXT PRIMARY KEY,
    repo_id      TEXT NOT NULL REFERENCES repos (id),
    author_login TEXT,
    message      TEXT NOT NULL,
    committed_at TEXT NOT NULL
);

CREATE TABLE pull_requests (
    id           TEXT PRIMARY KEY,
    repo_id      TEXT NOT NULL REFERENCES repos (id),
    number       INTEGER NOT NULL,
    title        TEXT NOT NULL,
    state        TEXT NOT NULL,
    author_login TEXT,
    body         TEXT,
    created_at   TEXT NOT NULL,
    merged_at    TEXT,
    files_synced INTEGER NOT NULL DEFAULT 0,
    html_url     TEXT
);

-- Changed files per PR, replaced wholesale on each sync. `patch` is NULL when the forge omits it
-- (binary or too large).
CREATE TABLE pr_files (
    pr_id     TEXT NOT NULL REFERENCES pull_requests (id),
    filename  TEXT NOT NULL,
    status    TEXT NOT NULL,
    additions INTEGER NOT NULL,
    deletions INTEGER NOT NULL,
    patch     TEXT,
    PRIMARY KEY (pr_id, filename)
);

-- Issues a PR closes, from the forge's authoritative references (not parsed from the body).
CREATE TABLE pr_closing_issues (
    pr_id        TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    PRIMARY KEY (pr_id, issue_number)
);

CREATE TABLE issues (
    id           TEXT PRIMARY KEY,
    repo_id      TEXT NOT NULL REFERENCES repos (id),
    number       INTEGER NOT NULL,
    title        TEXT NOT NULL,
    state        TEXT NOT NULL,
    author_login TEXT,
    body         TEXT,
    created_at   TEXT NOT NULL,
    closed_at    TEXT,
    labels       TEXT NOT NULL DEFAULT '',   -- newline-joined; a label can contain a comma
    html_url     TEXT
);

CREATE TABLE releases (
    id           TEXT PRIMARY KEY,
    repo_id      TEXT NOT NULL REFERENCES repos (id),
    tag          TEXT NOT NULL,
    name         TEXT,
    published_at TEXT
);

CREATE TABLE reviews (
    id             TEXT PRIMARY KEY,
    pr_id          TEXT NOT NULL REFERENCES pull_requests (id),
    reviewer_login TEXT,
    state          TEXT NOT NULL,
    submitted_at   TEXT
);

CREATE TABLE ci_runs (
    id           TEXT PRIMARY KEY,
    repo_id      TEXT NOT NULL REFERENCES repos (id),
    commit_sha   TEXT,
    status       TEXT NOT NULL,
    conclusion   TEXT,
    completed_at TEXT,
    html_url     TEXT,
    run_attempt  INTEGER               -- >1 means a re-run
);

-- Blob sha per indexed source file, so re-indexing skips unchanged files.
CREATE TABLE code_files (
    repo_id  TEXT NOT NULL REFERENCES repos (id),
    path     TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    PRIMARY KEY (repo_id, path)
);

-- AST metrics per source file. Stores computed metrics only, never file contents. `score` is
-- NULL exactly when the file was not analyzed.
CREATE TABLE file_health (
    repo_id   TEXT    NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    path      TEXT    NOT NULL,
    loc       INTEGER NOT NULL DEFAULT 0,
    analyzed  INTEGER NOT NULL DEFAULT 0 CHECK (analyzed IN (0, 1)),
    functions INTEGER NOT NULL DEFAULT 0,
    branches  INTEGER NOT NULL DEFAULT 0,
    score     INTEGER CHECK (score IS NULL OR score BETWEEN 1 AND 10),
    CHECK ((analyzed = 1) = (score IS NOT NULL)),
    PRIMARY KEY (repo_id, path)
);
CREATE INDEX file_health_repo_idx ON file_health (repo_id);

-- Declared direct dependencies, before any cross-repo resolution.
CREATE TABLE dependencies (
    repo_id     TEXT NOT NULL REFERENCES repos (id),
    ecosystem   TEXT NOT NULL,    -- 'cargo' | 'npm' | 'pip' | 'go'
    name        TEXT NOT NULL,
    version_req TEXT,             -- as written, or NULL (git/path deps)
    kind        TEXT NOT NULL,    -- 'normal' | 'dev' | 'build' | 'indirect'
    source      TEXT NOT NULL,    -- the manifest it came from, e.g. 'Cargo.toml'
    PRIMARY KEY (repo_id, source, name)
);

-- ---- organizing entities ----------------------------------------------------
-- `source` says where a row came from. A hand-defined row has the same shape as a derived one.

CREATE TABLE identities (
    id              TEXT PRIMARY KEY,
    canonical_login TEXT NOT NULL
);

-- Maps a per-repo login to a canonical identity.
CREATE TABLE identity_accounts (
    login       TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities (id)
);

CREATE TABLE teams (
    id     TEXT PRIMARY KEY,
    name   TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('github', 'manual'))
);

CREATE TABLE team_members (
    team_id     TEXT NOT NULL REFERENCES teams (id),
    identity_id TEXT NOT NULL REFERENCES identities (id),
    PRIMARY KEY (team_id, identity_id)
);

CREATE TABLE ownership (
    repo_id TEXT NOT NULL REFERENCES repos (id),
    team_id TEXT NOT NULL REFERENCES teams (id),
    PRIMARY KEY (repo_id, team_id)
);

CREATE TABLE sprints (
    id        TEXT PRIMARY KEY,
    team_id   TEXT REFERENCES teams (id),
    name      TEXT NOT NULL,
    starts_on TEXT,
    ends_on   TEXT,
    source    TEXT NOT NULL CHECK (source IN ('github', 'manual', 'jira'))
);

CREATE TABLE work_items (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    kind            TEXT NOT NULL,
    state           TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('github', 'manual', 'jira')),
    status_category TEXT               -- 'new' | 'indeterminate' | 'done'
);

CREATE TABLE sprint_work_items (
    sprint_id    TEXT NOT NULL REFERENCES sprints (id),
    work_item_id TEXT NOT NULL REFERENCES work_items (id),
    PRIMARY KEY (sprint_id, work_item_id)
);

-- Work item set frozen when a sprint is first seen active, so later scope changes cannot rewrite
-- what was committed.
CREATE TABLE sprint_commitments (
    sprint_id    TEXT NOT NULL,
    work_item_id TEXT NOT NULL,
    captured_on  TEXT NOT NULL,
    PRIMARY KEY (sprint_id, work_item_id)
);

-- When a work item first entered each status category. Basis for lead time and WIP.
CREATE TABLE work_item_status_history (
    work_item_id    TEXT NOT NULL,
    status_category TEXT NOT NULL,
    entered_at      TEXT NOT NULL,
    PRIMARY KEY (work_item_id, status_category)
);

-- Jira-specific fields for a work item or sprint (see the Jira ADR).
CREATE TABLE jira_issues (
    work_item_id    TEXT PRIMARY KEY REFERENCES work_items (id),
    tracker_id      TEXT NOT NULL,
    project         TEXT NOT NULL,
    issue_key       TEXT NOT NULL,
    assignee        TEXT,
    status_category TEXT,   -- 'new' | 'indeterminate' | 'done'
    url             TEXT,
    created_at      TEXT,
    updated_at      TEXT,
    resolved_at     TEXT,
    parent_key      TEXT
);
CREATE INDEX idx_jira_issues_key ON jira_issues (issue_key);
CREATE INDEX idx_jira_issues_project ON jira_issues (tracker_id, project);

CREATE TABLE jira_sprints (
    sprint_id    TEXT PRIMARY KEY REFERENCES sprints (id),
    tracker_id   TEXT NOT NULL,
    project      TEXT NOT NULL,
    state        TEXT,   -- 'active' | 'closed' | 'future'
    committed_at TEXT
);

-- Directed, typed edge between any two entities ('closes', 'references', 'depends', ...). The UNIQUE
-- constraint makes re-recording a link idempotent. See the work-item graph ADR.
CREATE TABLE links (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    src_kind TEXT NOT NULL,
    src_id   TEXT NOT NULL,
    dst_kind TEXT NOT NULL,
    dst_id   TEXT NOT NULL,
    relation TEXT NOT NULL,
    UNIQUE (src_kind, src_id, dst_kind, dst_id, relation)
);

-- ---- sync state and derived history ------------------------------------------

-- One row per (repo, entity): the incremental-fetch watermark (a timestamp, ETag or `pushed_at`) or a
-- backfill resume token. A sync reads it before fetching and advances it only when the window drained.
CREATE TABLE sync_cursors (
    repo_id TEXT NOT NULL,
    entity  TEXT NOT NULL,
    cursor  TEXT NOT NULL,
    PRIMARY KEY (repo_id, entity)
);

-- Daily metric snapshots in long format, so a new metric needs no migration. A same-day capture
-- overwrites. `captured_on` is supplied by the host; the core has no clock.
CREATE TABLE metric_snapshots (
    scope_kind  TEXT NOT NULL,
    scope_id    TEXT NOT NULL,
    captured_on TEXT NOT NULL,   -- YYYY-MM-DD
    metric_key  TEXT NOT NULL,
    value       INTEGER NOT NULL,
    PRIMARY KEY (scope_kind, scope_id, captured_on, metric_key)
);

-- Precomputed digest per scope; `body` is the serialized Digest JSON.
CREATE TABLE digests (
    scope_kind   TEXT NOT NULL,
    scope_id     TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    body         TEXT NOT NULL,
    PRIMARY KEY (scope_kind, scope_id)
);

-- ---- retrieval --------------------------------------------------------------
-- See the retrieval ADR (sqlite-vec plus FTS5, fused by rank).

-- One row per indexed chunk. `id` is the rowid of the matching vec0 and FTS rows.
CREATE TABLE embeddings (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_kind TEXT NOT NULL,
    ref_id   TEXT NOT NULL,
    chunk    TEXT NOT NULL
);
CREATE INDEX idx_embeddings_ref ON embeddings (ref_kind, ref_id);

-- Content hash per indexed entity, so unchanged text is not re-embedded.
CREATE TABLE indexed_text (
    ref_kind TEXT NOT NULL,
    ref_id   TEXT NOT NULL,
    hash     TEXT NOT NULL,
    PRIMARY KEY (ref_kind, ref_id)
);

-- The dimension must match core_embed's EMBED_DIM.
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    embedding float[768] distance_metric=cosine,
    ref_kind text
);

CREATE VIRTUAL TABLE fts_embeddings USING fts5(chunk);
