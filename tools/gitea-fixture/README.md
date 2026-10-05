# gitea fixture

A scripted local [Gitea](https://about.gitea.com/) that mirrors a public GitHub org (repos, issues,
pull requests, labels, milestones, releases, comments), so the Gitea backend and the desktop app have
realistic data to test against. No docker: the script downloads the static gitea binary and runs it
as a local process on SQLite.

## Requirements

`curl`, `jq` and `git`. A GitHub token is strongly recommended: set `GITHUB_TOKEN` or log in with the
`gh` CLI, which the script borrows. Anonymous calls are limited to 60 requests per hour.

## Use

```sh
cp .env.example .env    # optional
./mirror.sh all         # download and start gitea, then migrate every repo in the org
```

`up` starts gitea, creates the admin and mints a token. `mirror` creates users and migrates repos.
`all` runs both. Reruns are safe: existing repos and users are skipped. From the repo root:

```sh
just gitea-up
just gitea-mirror
just gitea-info       # reprint the base URL and admin token
just gitea-down       # stop, keep the data
just gitea-destroy    # stop and remove all state
```

To run the live test or the app against it:

```sh
export ORGONZOLA_FORGE_BASE_URL="http://localhost:3000/api/v1"
export ORGONZOLA_FORGE_TOKEN="$(cat tools/gitea-fixture/.gitea-token)"
```

## Configuration

Environment or `.env`. All have defaults.

| Variable               | Default                      | Meaning                                |
| ---------------------- | ---------------------------- | -------------------------------------- |
| `GITHUB_ORG`           | `tinygrad`                   | Public GitHub org to mirror.           |
| `GITHUB_TOKEN`         | `gh auth token`              | Token for listing and migration.       |
| `MAX_REPOS`            | `0` (all)                    | Cap on repos, newest-pushed first.     |
| `GITEA_VERSION`        | `1.22.6`                     | Gitea release to download.             |
| `GITEA_HTTP_PORT`      | `3000`                       | Port for the API and UI.               |
| `GITEA_ADMIN_USER`     | `fixture-admin`              | Admin login.                           |
| `GITEA_ADMIN_PASSWORD` | `fixture-admin-pw-change-me` | Admin password.                        |
| `GITEA_TARGET_ORG`     | same as `GITHUB_ORG`         | Gitea org the repos are migrated into. |

`tinygrad` is the default because it is active, about 22 repos, and mirrors in a few minutes.

## Limitations

- Authorship mapping is partial. Gitea links a migrated item to a local account only when it can
  match the login, and GitHub accounts cannot be recreated (no emails or passwords), so many items
  show the admin or a placeholder user. The script pre-creates accounts for members and contributors
  to link as many as it can.
- The instance is insecure on purpose: fixed secret key, admin token with all scopes, throwaway
  passwords. Local use only.
- Forks, stars, watchers and the social graph are not reproduced.

State lives in `.gitea-bin/` and `.gitea-token`, both gitignored.
