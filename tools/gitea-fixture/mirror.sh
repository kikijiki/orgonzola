#!/usr/bin/env bash
#
# Local Gitea that mirrors a public GitHub org, for forge integration tests and UI previews. Runs the
# static gitea binary as a plain process on SQLite; needs only curl, jq and git.
#
# Usage:
#   ./mirror.sh up                 # download + start gitea, create admin, mint token
#   ./mirror.sh mirror             # provision users + migrate every repo in the org
#   ./mirror.sh all                # up + mirror
#   ./mirror.sh info               # print base URL + admin token
#   ./mirror.sh logs               # follow the gitea log
#   ./mirror.sh down               # stop gitea, keep the data
#   ./mirror.sh destroy            # stop gitea and remove all state
#   ./mirror.sh --help
#
# Idempotent: reruns reuse the binary, config and db, and skip repos and users that exist. A
# GITHUB_TOKEN avoids anonymous rate limits; without one the script borrows `gh auth token`.
# Config comes from the environment or ./.env (see .env.example).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load ./.env if present; values already in the environment win.
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "$SCRIPT_DIR/.env"; set +a
fi

GITHUB_ORG="${GITHUB_ORG:-tinygrad}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
MAX_REPOS="${MAX_REPOS:-0}"

# Borrow the gh CLI's token if none was given.
if [[ -z "$GITHUB_TOKEN" ]] && command -v gh >/dev/null 2>&1; then
  GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
fi

GITEA_VERSION="${GITEA_VERSION:-1.22.6}"
GITEA_HTTP_PORT="${GITEA_HTTP_PORT:-3000}"
GITEA_ADMIN_USER="${GITEA_ADMIN_USER:-fixture-admin}"
GITEA_ADMIN_PASSWORD="${GITEA_ADMIN_PASSWORD:-fixture-admin-pw-change-me}"
GITEA_ADMIN_EMAIL="${GITEA_ADMIN_EMAIL:-fixture-admin@example.com}"
GITEA_TARGET_ORG="${GITEA_TARGET_ORG:-$GITHUB_ORG}"
GITEA_ROOT_URL="${GITEA_ROOT_URL:-http://localhost:${GITEA_HTTP_PORT}/}"

# The work dir holds the binary, config, db, log and pid. It is gitignored; `destroy` removes it.
GITEA_BASE_URL="http://localhost:${GITEA_HTTP_PORT}"
GITEA_API="${GITEA_BASE_URL}/api/v1"
TOKEN_FILE="$SCRIPT_DIR/.gitea-token"
WORK_DIR="$SCRIPT_DIR/.gitea-bin"
BIN_PATH="$WORK_DIR/gitea"
PID_FILE="$WORK_DIR/gitea.pid"

GITHUB_API="https://api.github.com"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"; }

# Paginated GitHub GET. Prints each array element as one JSON line.
gh_get_paged() {
  local url="$1" auth=()
  [[ -n "$GITHUB_TOKEN" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
  local page=1
  while :; do
    local resp
    resp="$(curl -fsSL "${auth[@]}" -H "Accept: application/vnd.github+json" \
      "${url}?per_page=100&page=${page}")" || die "GitHub API call failed: $url (page $page)"
    local count
    count="$(jq 'length' <<<"$resp")"
    [[ "$count" -eq 0 ]] && break
    jq -c '.[]' <<<"$resp"
    [[ "$count" -lt 100 ]] && break
    page=$((page + 1))
  done
}

# Gitea API call: gitea_api METHOD PATH [json-body]. Echoes the body, sets GITEA_LAST_CODE.
GITEA_LAST_CODE=""
gitea_api() {
  local method="$1" path="$2" body="${3:-}"
  local token; token="$(cat "$TOKEN_FILE" 2>/dev/null || true)"
  [[ -n "$token" ]] || die "no gitea admin token yet - run '$0 up' first"
  local args=(-sS -X "$method" -H "Authorization: token $token"
              -H "Content-Type: application/json" -H "Accept: application/json"
              -w '\n%{http_code}' "${GITEA_API}${path}")
  [[ -n "$body" ]] && args+=(-d "$body")
  local out; out="$(curl "${args[@]}")"
  GITEA_LAST_CODE="${out##*$'\n'}"
  printf '%s' "${out%$'\n'*}"
}

# Run gitea with HOME and XDG_CONFIG_HOME inside the work dir. Gitea runs `git config --global` on
# init, which fails against a read-only config (e.g. a nix home-manager symlink).
gitea_run() {
  HOME="$WORK_DIR/home" XDG_CONFIG_HOME="$WORK_DIR/home/.config" GIT_CONFIG_NOSYSTEM=1 \
    GITEA_WORK_DIR="$WORK_DIR" GITEA_CUSTOM="$WORK_DIR/custom" \
    "$BIN_PATH" "$@"
}

# True if the pid file names a live process.
gitea_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

cmd_up() {
  need curl; need jq; need git
  mkdir -p "$WORK_DIR/custom/conf" "$WORK_DIR/data" "$WORK_DIR/log" "$WORK_DIR/home/.config/git"

  if [[ ! -x "$BIN_PATH" ]]; then
    local os arch
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    case "$(uname -m)" in
      x86_64|amd64) arch=amd64 ;;
      aarch64|arm64) arch=arm64 ;;
      *) die "unsupported arch: $(uname -m) (gitea publishes amd64 and arm64 binaries)" ;;
    esac
    log "downloading gitea ${GITEA_VERSION} (${os}-${arch})"
    curl -fsSL -o "$BIN_PATH" \
      "https://dl.gitea.com/gitea/${GITEA_VERSION}/gitea-${GITEA_VERSION}-${os}-${arch}" \
      || die "failed to download the gitea binary"
    chmod +x "$BIN_PATH"
  fi

  # Installer locked so gitea boots straight into a working server. The secrets are throwaways.
  cat > "$WORK_DIR/custom/conf/app.ini" <<INI
APP_NAME = orgonzola-fixture
RUN_MODE = prod
[server]
HTTP_PORT = ${GITEA_HTTP_PORT}
ROOT_URL = ${GITEA_ROOT_URL}
DISABLE_SSH = true
[database]
DB_TYPE = sqlite3
PATH = ${WORK_DIR}/data/gitea.db
[security]
INSTALL_LOCK = true
SECRET_KEY = orgonzola-fixture-secret-not-for-prod
INTERNAL_TOKEN = orgonzolafixtureinternaltoken
[service]
DISABLE_REGISTRATION = true
[log]
LEVEL = warn
INI

  if gitea_running; then
    log "gitea already running (pid $(cat "$PID_FILE")) on port $GITEA_HTTP_PORT"
  else
    log "starting gitea on port $GITEA_HTTP_PORT (log: $WORK_DIR/log/gitea.log)"
    HOME="$WORK_DIR/home" XDG_CONFIG_HOME="$WORK_DIR/home/.config" GIT_CONFIG_NOSYSTEM=1 \
      GITEA_WORK_DIR="$WORK_DIR" GITEA_CUSTOM="$WORK_DIR/custom" \
      nohup "$BIN_PATH" web --config "$WORK_DIR/custom/conf/app.ini" \
        > "$WORK_DIR/log/gitea.log" 2>&1 &
    echo $! > "$PID_FILE"
  fi

  log "waiting for the gitea API to come up..."
  local tries=0
  until curl -fsS "${GITEA_BASE_URL}/api/healthz" >/dev/null 2>&1; do
    tries=$((tries + 1))
    [[ $tries -gt 60 ]] && die "gitea did not come up in time (check: $0 logs)"
    sleep 2
  done
  log "gitea API is up at ${GITEA_API}"

  ensure_admin
  mint_token
  log "admin ready. token cached in $TOKEN_FILE"
  cmd_info
}

# Create the admin user via the gitea CLI. An already-existing user counts as success.
ensure_admin() {
  log "ensuring admin user '$GITEA_ADMIN_USER' exists"
  local out
  if out="$(gitea_run admin user create \
        --username "$GITEA_ADMIN_USER" \
        --password "$GITEA_ADMIN_PASSWORD" \
        --email "$GITEA_ADMIN_EMAIL" \
        --admin --must-change-password=false 2>&1)"; then
    log "admin user created"
  else
    if grep -qiE 'already exists|user already' <<<"$out"; then
      log "admin user already exists (ok)"
    else
      die "failed to create admin user: $out"
    fi
  fi
}

# Mint an admin API token with all scopes and cache it.
mint_token() {
  log "minting an admin API token"
  # The CLI prints the token only on creation and names must be unique, so use a timestamped name.
  local name="orgonzola-fixture-$(date +%s)"
  local out
  out="$(gitea_run admin user generate-access-token \
      --username "$GITEA_ADMIN_USER" --token-name "$name" --scopes all 2>&1)" \
    || die "failed to mint token: $out"
  local token; token="$(grep -oE '[0-9a-f]{40}' <<<"$out" | head -n1)"
  [[ -n "$token" ]] || die "could not parse a token from CLI output: $out"
  printf '%s' "$token" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
}

# Create the target org if missing.
ensure_org() {
  log "ensuring target gitea org '$GITEA_TARGET_ORG' exists"
  gitea_api GET "/orgs/${GITEA_TARGET_ORG}" >/dev/null
  if [[ "$GITEA_LAST_CODE" == "200" ]]; then
    log "org already exists (ok)"; return
  fi
  local body; body="$(jq -nc --arg u "$GITEA_TARGET_ORG" '{username:$u, visibility:"public"}')"
  gitea_api POST "/orgs" "$body" >/dev/null
  [[ "$GITEA_LAST_CODE" =~ ^20 ]] || die "failed to create org (HTTP $GITEA_LAST_CODE)"
  log "org created"
}

# Create a Gitea user for a GitHub login so migrated authorship can map to it. Passwords are a fixed
# throwaway; these are fixture accounts.
ensure_user() {
  local login="$1"
  # Normalize to a valid Gitea username: lowercase, disallowed chars to '-', trim edge separators.
  # printf, not a herestring, so no trailing newline becomes a stray '-'.
  local uname
  uname="$(printf '%s' "$login" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-')"
  uname="${uname#"${uname%%[!._-]*}"}"   # strip leading . _ -
  uname="${uname%"${uname##*[!._-]}"}"   # strip trailing . _ -
  [[ -z "$uname" ]] && return 0
  gitea_api GET "/users/${uname}" >/dev/null
  [[ "$GITEA_LAST_CODE" == "200" ]] && return 0
  local body
  body="$(jq -nc --arg u "$uname" --arg e "${uname}@users.noreply.fixture.local" \
      '{username:$u, email:$e, password:"fixture-user-pw-0000", must_change_password:false}')"
  gitea_api POST "/admin/users" "$body" >/dev/null
  if [[ "$GITEA_LAST_CODE" =~ ^20 ]]; then
    printf '.' >&2
  elif [[ "$GITEA_LAST_CODE" == "422" ]]; then
    : # already exists - fine
  else
    warn "could not create user '$uname' (HTTP $GITEA_LAST_CODE)"
  fi
}

# Create users for the org's members and the given repos' contributors. Gitea's migration maps
# authors to existing accounts by login, else to a placeholder ghost; it creates none itself.
# Org membership is often hidden, so contributors are the reliable source.
provision_users() {
  log "provisioning gitea users from github org '$GITHUB_ORG' (members + repo contributors)"
  local seen=" " login

  add_login() {
    local l="$1"
    [[ -z "$l" || "$l" == "null" ]] && return 0
    [[ "$seen" == *" $l "* ]] && return 0
    seen+="$l "
    ensure_user "$l" || true
  }

  # Org members; empty for orgs that hide membership.
  while IFS= read -r login; do add_login "$login"; done < <(
    gh_get_paged "${GITHUB_API}/orgs/${GITHUB_ORG}/members" 2>/dev/null | jq -r '.login' || true)

  local name
  for name in "$@"; do
    while IFS= read -r login; do add_login "$login"; done < <(
      gh_get_paged "${GITHUB_API}/repos/${GITHUB_ORG}/${name}/contributors" 2>/dev/null \
        | jq -r 'select(.type != "Bot") | .login' || true)
  done

  printf '\n' >&2
  log "user provisioning done"
}

# Migrate one GitHub repo into the target org. With a token the API also imports issues, PRs, labels,
# milestones, releases and comments.
migrate_repo() {
  local name="$1" clone_url="$2"
  gitea_api GET "/repos/${GITEA_TARGET_ORG}/${name}" >/dev/null
  if [[ "$GITEA_LAST_CODE" == "200" ]]; then
    log "  $name already mirrored (skip)"; return
  fi
  log "  migrating $name"
  local body
  body="$(jq -nc \
    --arg clone "$clone_url" \
    --arg owner "$GITEA_TARGET_ORG" \
    --arg name "$name" \
    --arg tok "$GITHUB_TOKEN" \
    '{
       clone_addr: $clone,
       repo_owner: $owner,
       repo_name:  $name,
       service:    "github",
       mirror:     false,
       private:    false,
       wiki:        true,
       issues:      true,
       pull_requests: true,
       labels:      true,
       milestones:  true,
       releases:    true,
       auth_token:  $tok
     }')"
  gitea_api POST "/repos/migrate" "$body" >/dev/null
  if [[ "$GITEA_LAST_CODE" =~ ^20 ]]; then
    log "    done"
  elif [[ "$GITEA_LAST_CODE" == "409" ]]; then
    log "    already exists (ok)"
  else
    warn "    migration of $name failed (HTTP $GITEA_LAST_CODE) - continuing"
  fi
}

cmd_mirror() {
  need curl; need jq
  [[ -f "$TOKEN_FILE" ]] || die "no admin token - run '$0 up' first"
  [[ -n "$GITHUB_TOKEN" ]] || warn "no GITHUB_TOKEN set - GitHub calls are anonymous and rate-limited"

  ensure_org

  log "listing repos for github org '$GITHUB_ORG'"
  # Newest-pushed first, so a MAX_REPOS cap keeps the most active repos.
  local repos_json
  repos_json="$(gh_get_paged "${GITHUB_API}/orgs/${GITHUB_ORG}/repos" \
    | jq -sc 'sort_by(.pushed_at) | reverse')"
  local total; total="$(jq 'length' <<<"$repos_json")"
  [[ "$total" -eq 0 ]] && die "no repos found for org '$GITHUB_ORG' (typo? private org? rate limit?)"

  if [[ "$MAX_REPOS" -gt 0 && "$MAX_REPOS" -lt "$total" ]]; then
    log "capping to MAX_REPOS=$MAX_REPOS of $total repos"
    repos_json="$(jq -c ".[:$MAX_REPOS]" <<<"$repos_json")"
  else
    log "mirroring all $total repos"
  fi

  # Create users before migrating so authorship can map to them.
  local repo_names=()
  while IFS= read -r n; do [[ -n "$n" ]] && repo_names+=("$n"); done < <(
    jq -r '.[].name' <<<"$repos_json")
  provision_users "${repo_names[@]}"

  while IFS=$'\t' read -r name clone; do
    [[ -z "$name" ]] && continue
    migrate_repo "$name" "$clone"
  done < <(jq -r '.[] | [.name, .clone_url] | @tsv' <<<"$repos_json")

  log "mirror complete"
  cmd_info
}

cmd_all() { cmd_up; cmd_mirror; }

cmd_info() {
  local token; token="$(cat "$TOKEN_FILE" 2>/dev/null || echo '<run "up" first>')"
  cat >&2 <<EOF

  ----------------------------------------------------------------------
  gitea fixture is ready - point tests / the desktop app at it:

    forge base URL : ${GITEA_API}
    web UI         : ${GITEA_BASE_URL}
    admin user     : ${GITEA_ADMIN_USER} / ${GITEA_ADMIN_PASSWORD}
    admin token    : ${token}

  example (the app / tests read a base URL + token):
    export ORGONZOLA_FORGE_BASE_URL="${GITEA_API}"
    export ORGONZOLA_FORGE_TOKEN="${token}"

  mirrored github org : ${GITHUB_ORG}  ->  gitea org : ${GITEA_TARGET_ORG}
  ----------------------------------------------------------------------
EOF
}

cmd_logs() {
  [[ -f "$WORK_DIR/log/gitea.log" ]] || die "no log yet - run '$0 up' first"
  tail -f "$WORK_DIR/log/gitea.log"
}

# Stop the server, keep the data. A later `up` resumes with the mirrored org intact.
cmd_down() {
  log "stopping gitea (data kept; 'up' resumes it, 'destroy' wipes it)"
  if gitea_running; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PID_FILE"
  log "stopped. data preserved in $WORK_DIR/data"
}

# Stop the server and remove all state (binary, db, config, token).
cmd_destroy() {
  cmd_down
  log "removing the fixture work dir + token"
  # Only ever remove our own .gitea-bin, never an empty or foreign path.
  if [[ -n "$WORK_DIR" && "$WORK_DIR" == "$SCRIPT_DIR/.gitea-bin" && -d "$WORK_DIR" ]]; then
    rm -rf "$WORK_DIR"
  fi
  rm -f "$TOKEN_FILE"
  log "done. all fixture state removed."
}

usage() {
  # Print the leading comment block.
  awk 'NR>1 && /^set -euo pipefail/{exit} NR>1{sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"
}

case "${1:-}" in
  up)            cmd_up ;;
  mirror)        cmd_mirror ;;
  all)           cmd_all ;;
  info)          cmd_info ;;
  logs)          cmd_logs ;;
  down)          cmd_down ;;
  destroy)       cmd_destroy ;;
  -h|--help|"")  usage ;;
  *)             die "unknown command: $1 (try --help)" ;;
esac
