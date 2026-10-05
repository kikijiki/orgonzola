# orgonzola task runner. Run recipes inside `nix develop`; `just` with no recipe lists them.

# default: show the recipe list
default:
    @just --list

# ---- rust core (host-agnostic) ----------------------------------------------

# build the whole cargo workspace
build:
    cargo build --workspace

# fast type-check the workspace (runs fmt-check first)
#
# `--features http` compiles core-github's transport and core-forge-gitea's live test, which are
# otherwise built only because hosts/desktop happens to enable the feature. `core-embed/fastembed`
# is cheap here: `check` does not link, so ort-sys never downloads onnxruntime.
[doc("fast type-check the workspace (runs fmt-check first)")]
check: fmt-check
    cargo check --workspace --features http,core-embed/fastembed

# unit tests over the workspace, no host or display needed
#
# The second command builds core-github alone. Under `--workspace`, shared deps carry every feature
# any member enables (hosts/desktop adds `json` and `stream` to reqwest), so core-github could rely on
# a feature it never declares. fastembed stays off: `test` links, and would download onnxruntime.
[doc("run the unit tests over the core crates (no host, no display)")]
test:
    cargo test --workspace --features http
    cargo test -p core-github --features http

# format every crate, in place
fmt:
    cargo fmt --all

# verify formatting without writing; `check` runs it first
#
# No `--features`: rustfmt only parses, so feature-gated code is formatted either way.
[doc("verify formatting without writing (the read-only half of `fmt`)")]
fmt-check:
    cargo fmt --all -- --check

# lint the workspace, warnings as errors; same features as `check`
[doc("lint the workspace, warnings as errors")]
clippy:
    cargo clippy --workspace --all-targets --features http,core-embed/fastembed -- -D warnings

# run the `#[ignore]`d live tests against real Jira and GitHub fixtures
#
# Needs ORGONZOLA_GITHUB_TOKEN, ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN.
[doc("run the live Jira and GitHub tests (needs credentials in the environment)")]
live-test:
    cargo test -p core-jira --features http -- --ignored --nocapture
    cargo test -p core-forge-github --features http -- --ignored --nocapture

# ---- typescript ui (react + vite + tailwind + shadcn) -----------------------

# install UI deps
ui-install:
    cd ui && pnpm install

# run the UI dev server standalone (Vite)
ui-dev:
    cd ui && pnpm dev

# type-check + lint the UI
ui-check:
    cd ui && pnpm run typecheck && pnpm run lint && pnpm run test

# build the UI bundle
ui-build:
    cd ui && pnpm build

# ---- desktop app (the tauri shell owns the long-running process) ------------

# launch the app: fetch the embedding and LLM models if missing, then `cargo tauri dev`
#
# The LLM runs on llama.cpp with Vulkan (Intel Arc, NVIDIA, AMD) and falls back to CPU. The build
# needs the Vulkan SDK, glslc, cmake and libclang, which the nix devshell provides.
[doc("launch the app (fetches models if missing, then cargo tauri dev)")]
run FILE="Qwen3-0.6B-Q4_K_M.gguf": fetch-model (fetch-llm-model "unsloth/Qwen3-0.6B-GGUF" FILE)
    ORGONZOLA_LLM_DIR="$PWD/hosts/desktop/resources/models/llm" ORGONZOLA_LLM_GGUF="{{FILE}}" \
        cargo tauri dev --features fastembed

# build a release bundle, fetching the models first so they ship in it
bundle FILE="Qwen3-0.6B-Q4_K_M.gguf": fetch-model (fetch-llm-model "unsloth/Qwen3-0.6B-GGUF" FILE)
    cargo tauri build --features fastembed

# fetch the embedding and reranker models into the gitignored resources dir. Idempotent.
# Uses the int8 ONNX graphs to keep the bundle small (~200 MB vs ~800 MB).
[doc("fetch the embedding and reranker models if missing")]
fetch-model:
    #!/usr/bin/env bash
    set -euo pipefail
    # local name -> path in the HF repo. `from_path` expects a flat dir with the graph named model.onnx.
    local_names=(model.onnx tokenizer.json config.json special_tokens_map.json tokenizer_config.json)
    remote_paths=(onnx/model_quantized.onnx tokenizer.json config.json special_tokens_map.json tokenizer_config.json)
    fetch_one() {
        local dir="hosts/desktop/resources/models/$1"
        local base="https://huggingface.co/$2/resolve/main"
        local missing=0
        for f in "${local_names[@]}"; do [ -f "$dir/$f" ] || missing=1; done
        if [ "$missing" -eq 0 ]; then echo "model already present in $dir"; return 0; fi
        mkdir -p "$dir"
        for i in "${!local_names[@]}"; do
            echo "fetching $1/${local_names[$i]}"
            curl -fsSL "$base/${remote_paths[$i]}" -o "$dir/${local_names[$i]}"
        done
        echo "model ready in $dir"
    }
    fetch_one jina-embeddings-v2-base-code jinaai/jina-embeddings-v2-base-code
    fetch_one jina-reranker-v1-turbo-en jinaai/jina-reranker-v1-turbo-en

# fetch a quantized GGUF for the local LLM into the gitignored resources dir. Idempotent.
# Override REPO and FILE for another model.
[doc("fetch a GGUF for the local LLM if missing")]
fetch-llm-model REPO="unsloth/Qwen3-0.6B-GGUF" FILE="Qwen3-0.6B-Q4_K_M.gguf":
    #!/usr/bin/env bash
    set -euo pipefail
    dir="hosts/desktop/resources/models/llm"
    if [ -f "$dir/{{FILE}}" ]; then echo "llm model already present: $dir/{{FILE}}"; exit 0; fi
    mkdir -p "$dir"
    echo "fetching {{REPO}}/{{FILE}}"
    curl -fsSL "https://huggingface.co/{{REPO}}/resolve/main/{{FILE}}" -o "$dir/{{FILE}}"
    echo "llm model ready: $dir/{{FILE}}"

# ---- docs hygiene -----------------------------------------------------------

# markdown files that docs-check and docs-fix operate on; `--others` includes uncommitted files
[private]
_docs-files:
    @git ls-files --cached --others --exclude-standard '*.md'

# check docs without writing: prettier formatting, links (lychee), ASCII only
docs-check:
    #!/usr/bin/env bash
    set -uo pipefail
    rc=0
    files=$(just _docs-files)
    echo "-- formatting --"
    prettier --check $files || rc=1
    echo "-- links --"
    lychee --no-progress $files || rc=1
    echo "-- ascii (no em-dashes / non-ascii) --"
    if grep -nP '[^\x00-\x7F]' $files; then
        echo "non-ASCII characters found (see above) - replace with ASCII equivalents"
        rc=1
    fi
    exit $rc

# reformat every markdown file in place (prettier)
docs-fix:
    prettier --write $(just _docs-files)

# ---- gitea fixture ----------------------------------------------------------
# Local Gitea mirroring a public GitHub org (default: tinygrad), run as a plain process (no docker).
# Needs curl, jq, git, and GITHUB_TOKEN or `gh auth token`. See tools/gitea-fixture/README.md.

# download + start gitea, create the admin, mint an API token
gitea-up:
    tools/gitea-fixture/mirror.sh up

# provision users + migrate the org's repos into gitea
gitea-mirror:
    tools/gitea-fixture/mirror.sh mirror

# stop gitea but keep the data (gitea-up resumes the same mirror, no re-migration)
gitea-down:
    tools/gitea-fixture/mirror.sh down

# stop gitea and remove all fixture state (db, binary, token) - a from-scratch reset
gitea-destroy:
    tools/gitea-fixture/mirror.sh destroy

# print the gitea base URL + admin token (for tests / the app)
gitea-info:
    tools/gitea-fixture/mirror.sh info

# ---- storage ----------------------------------------------------------------
# A full build is several GB (llama.cpp compiles from source). Disk use falls into three buckets:
#   rebuildable   cargo target dirs, ui/node_modules, fastembed cache. `storage-clean` removes them.
#   re-fetchable  models under hosts/desktop/resources/models, gitea fixture state. Never auto-removed.
#   user data     the app database and downloaded LLM models in the OS app-data dir. No recipe touches it.

# report disk use by bucket. Read-only.
storage:
    #!/usr/bin/env bash
    set -uo pipefail
    shopt -s nullglob
    sum=0
    row() {
        if [ -e "$1" ]; then
            b=$(du -sb "$1" 2>/dev/null | cut -f1)
            sum=$((sum + b))
            printf '  %9s  %s\n' "$(numfmt --to=iec --suffix=B "$b")" "$1"
        else
            printf '  %9s  %s (absent)\n' "-" "$1"
        fi
    }
    root=$(git rev-parse --show-toplevel)
    repo_target="${CARGO_TARGET_DIR:-$root/target}"
    if [ "$(uname)" = "Darwin" ]; then
        app_data="$HOME/Library/Application Support/com.kikijiki.orgonzola"
    else
        app_data="${XDG_DATA_HOME:-$HOME/.local/share}/com.kikijiki.orgonzola"
    fi

    echo "rebuildable - 'just storage-clean' removes these:"
    sum=0
    row "$repo_target"
    row "$root/ui/node_modules"
    for c in $(find "$root" -type d -name .fastembed_cache -not -path '*/target/*' 2>/dev/null); do row "$c"; done
    rebuildable=$sum

    echo
    echo "re-fetchable - gitignored downloads, NOT removed by any recipe here:"
    sum=0
    row "$root/hosts/desktop/resources/models"
    row "$root/tools/gitea-fixture/.gitea-bin"
    echo "             models come back with 'just fetch-model' / 'just run' (large download);"
    echo "             gitea state is owned by 'just gitea-destroy' (full re-migration to restore)."

    echo
    echo "user data - the local-first store. NEVER touched by a cleanup recipe:"
    sum=0
    row "$app_data"
    echo "             holds orgonzola.db plus downloaded LLM models. Not a build artifact."

    echo
    printf '  reclaimable now: %s\n' "$(numfmt --to=iec --suffix=B "$rebuildable")"
    df -h "$root" | tail -1 | awk '{printf "  filesystem %s: %s used of %s, %s free\n", $6, $3, $2, $4}'

# remove rebuildable artifacts: target dir, ui/node_modules, .fastembed_cache
#
# Never removes the app-data dir, fetched models, or the gitea fixture.
[doc("reclaim rebuildable artifacts (target, node_modules, fastembed cache)")]
storage-clean:
    #!/usr/bin/env bash
    set -uo pipefail
    root=$(git rev-parse --show-toplevel)
    drop() {
        [ -e "$1" ] || { printf '  skip     %s (absent)\n' "$1"; return; }
        b=$(du -sb "$1" 2>/dev/null | cut -f1)
        rm -rf "$1"
        printf '  removed  %9s  %s\n' "$(numfmt --to=iec --suffix=B "$b")" "$1"
    }
    drop "${CARGO_TARGET_DIR:-$root/target}"
    drop "$root/ui/node_modules"
    for c in $(find "$root" -type d -name .fastembed_cache 2>/dev/null); do drop "$c"; done
    echo "the app database, the fetched models and the gitea fixture were left alone - see 'just storage'"

# ---- misc -------------------------------------------------------------------

# count lines of code by language
loc:
    tokei
