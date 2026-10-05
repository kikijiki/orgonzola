//! Dependency graph: pure manifest parsing into a normalized list of declared direct
//! dependencies. Supports `Cargo.toml`, `package.json`, `requirements.txt`, `pyproject.toml` and
//! `go.mod`. A given manifest text always yields the same edge set. Fetching and persisting live
//! in `core-sync` / `core-store`.

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("failed to parse Cargo.toml: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("failed to parse package.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unrecognized manifest file: {0}")]
    UnknownManifest(String),
}

/// A declared direct dependency of a repo. `version_req` is the requirement as written, or
/// `None` when the manifest gives none (e.g. a git/path Cargo dependency).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dependency {
    /// "cargo" | "npm" | "pip" | "go".
    pub ecosystem: String,
    pub name: String,
    pub version_req: Option<String>,
    /// "normal" | "dev" | "build" | "indirect".
    pub kind: String,
}

/// Dispatch on the manifest file name; `UnknownManifest` for a name we do not handle.
pub fn parse_manifest(file_name: &str, text: &str) -> Result<Vec<Dependency>, GraphError> {
    match file_name {
        "Cargo.toml" => parse_cargo_toml(text),
        "package.json" => parse_package_json(text),
        "requirements.txt" => Ok(parse_requirements_txt(text)),
        "pyproject.toml" => parse_pyproject_toml(text),
        "go.mod" => Ok(parse_go_mod(text)),
        other => Err(GraphError::UnknownManifest(other.to_string())),
    }
}

/// Parse a `Cargo.toml`: `[dependencies]`, `[dev-dependencies]` and `[build-dependencies]`. A
/// value is a version string or a table; a table with no `version` (git/path) yields
/// `version_req = None`.
pub fn parse_cargo_toml(text: &str) -> Result<Vec<Dependency>, GraphError> {
    let value: toml::Value = toml::from_str(text)?;
    let mut deps = Vec::new();
    for (table, kind) in [
        ("dependencies", "normal"),
        ("dev-dependencies", "dev"),
        ("build-dependencies", "build"),
    ] {
        let Some(toml::Value::Table(entries)) = value.get(table) else {
            continue;
        };
        for (name, spec) in entries {
            let version_req = match spec {
                toml::Value::String(v) => Some(v.clone()),
                toml::Value::Table(t) => {
                    t.get("version").and_then(|v| v.as_str()).map(String::from)
                }
                _ => None,
            };
            deps.push(Dependency {
                ecosystem: "cargo".to_string(),
                name: name.clone(),
                version_req,
                kind: kind.to_string(),
            });
        }
    }
    Ok(deps)
}

/// Parse a `package.json`: `dependencies` and `devDependencies`.
pub fn parse_package_json(text: &str) -> Result<Vec<Dependency>, GraphError> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    let mut deps = Vec::new();
    for (field, kind) in [("dependencies", "normal"), ("devDependencies", "dev")] {
        let Some(serde_json::Value::Object(entries)) = value.get(field) else {
            continue;
        };
        for (name, spec) in entries {
            deps.push(Dependency {
                ecosystem: "npm".to_string(),
                name: name.clone(),
                version_req: spec.as_str().map(String::from),
                kind: kind.to_string(),
            });
        }
    }
    Ok(deps)
}

/// Split a PEP 508 requirement into a package name (the leading run of `A-Za-z0-9._-`) and the
/// rest. Handles `requests==2.0`, `flask>=1,<2`, `pkg[extra]>=1`, `name @ url`. `None` for an
/// empty or comment line.
fn split_pep508(line: &str) -> Option<(String, Option<String>)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') || line.starts_with("-") {
        return None; // blank, comment, or a pip option line (-r, --hash, ...)
    }
    let name_end = line
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
        .unwrap_or(line.len());
    let name = &line[..name_end];
    if name.is_empty() {
        return None;
    }
    let rest = line[name_end..].trim();
    let version_req = if rest.is_empty() {
        None
    } else {
        Some(rest.to_string()) // keeps any extras + specifiers, e.g. "[security]>=2.0"
    };
    Some((name.to_string(), version_req))
}

/// Parse a `requirements.txt`: one PEP 508 requirement per line; `#` comments and pip option
/// lines are skipped.
pub fn parse_requirements_txt(text: &str) -> Vec<Dependency> {
    text.lines()
        .filter_map(|line| {
            let line = line.split('#').next().unwrap_or(line);
            split_pep508(line).map(|(name, version_req)| Dependency {
                ecosystem: "pip".to_string(),
                name,
                version_req,
                kind: "normal".to_string(),
            })
        })
        .collect()
}

/// Parse a `pyproject.toml`: PEP 621 `[project].dependencies` as `normal`,
/// `[project.optional-dependencies]` groups as `dev`.
pub fn parse_pyproject_toml(text: &str) -> Result<Vec<Dependency>, GraphError> {
    let value: toml::Value = toml::from_str(text)?;
    let mut deps = Vec::new();

    let push_list = |deps: &mut Vec<Dependency>, list: &toml::Value, kind: &str| {
        if let toml::Value::Array(items) = list {
            for item in items {
                if let Some((name, version_req)) = item.as_str().and_then(split_pep508) {
                    deps.push(Dependency {
                        ecosystem: "pip".to_string(),
                        name,
                        version_req,
                        kind: kind.to_string(),
                    });
                }
            }
        }
    };

    if let Some(project) = value.get("project") {
        if let Some(list) = project.get("dependencies") {
            push_list(&mut deps, list, "normal");
        }
        // optional-dependencies is a table of group -> list; all are treated as dev extras.
        if let Some(toml::Value::Table(groups)) = project.get("optional-dependencies") {
            for list in groups.values() {
                push_list(&mut deps, list, "dev");
            }
        }
    }
    Ok(deps)
}

/// Parse a `go.mod`: `require` directives in single-line and block form. A `// indirect` comment
/// marks a transitive dep (kind `indirect`); the rest are `normal`.
pub fn parse_go_mod(text: &str) -> Vec<Dependency> {
    let mut deps = Vec::new();
    let mut in_block = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if in_block {
            if line.starts_with(')') {
                in_block = false;
                continue;
            }
            if let Some(dep) = parse_go_require_entry(line) {
                deps.push(dep);
            }
        } else if line == "require (" || line == "require(" {
            in_block = true;
        } else if let Some(rest) = line.strip_prefix("require ") {
            if let Some(dep) = parse_go_require_entry(rest.trim()) {
                deps.push(dep);
            }
        }
    }
    deps
}

/// One `module version [// indirect]` entry from a go.mod require directive.
fn parse_go_require_entry(entry: &str) -> Option<Dependency> {
    let indirect = entry.contains("// indirect");
    let code = entry.split("//").next().unwrap_or(entry).trim();
    let mut parts = code.split_whitespace();
    let name = parts.next()?.to_string();
    let version_req = parts.next().map(String::from);
    Some(Dependency {
        ecosystem: "go".to_string(),
        name,
        version_req,
        kind: if indirect { "indirect" } else { "normal" }.to_string(),
    })
}

/// Resolve a declared dependency to the `owner/repo` of its source on `host`, when the
/// dependency encodes one: a `go` module path under that host, or an `npm`/`pip` git dependency
/// whose requirement is a URL on that host. `host` is as the manifest would write it
/// (`github.com`, `gitea.example.com`). `None` for registry dependencies named by package only
/// (resolving those would need a registry lookup). Case is preserved.
pub fn repo_on_host(
    ecosystem: &str,
    name: &str,
    version_req: Option<&str>,
    host: &str,
) -> Option<String> {
    if host.is_empty() {
        return None;
    }
    // go: the module path is the repo path, e.g. github.com/acme/widget[/v2][/subpkg].
    if ecosystem == "go" {
        return owner_repo_on_host(name, host);
    }
    // npm / pip git deps: the requirement carries the URL or `github:` shorthand.
    version_req.and_then(|req| owner_repo_on_host(req, host))
}

/// The one host with a package-manager shorthand we can resolve. Gitea and Forgejo have none, so
/// a self-hosted host is reachable only by full URL or module path.
const GITHUB_HOST: &str = "github.com";

/// Pull `owner/repo` out of a string holding a path on `host`, or, for `github.com` only, a
/// `github:owner/repo` shorthand. Takes the first two path segments after the host, stripping a
/// `.git` suffix and any `#fragment` / `?query`. `None` if no repo is found.
fn owner_repo_on_host(s: &str, host: &str) -> Option<String> {
    let s = s.trim();
    let rest = match s.strip_prefix("github:").filter(|_| host == GITHUB_HOST) {
        Some(shorthand) => shorthand,
        None => path_after_host(s, host)?,
    };
    let mut segs = rest.split('/').filter(|p| !p.is_empty());
    let owner = segs.next()?;
    let repo = segs.next()?;
    // Trim a trailing `.git` and any `#fragment` / `?query`.
    let repo = repo
        .split(['#', '?'])
        .next()
        .unwrap_or(repo)
        .trim_end_matches(".git");
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{owner}/{repo}"))
}

/// The path that follows an occurrence of `host` in `s`, when that occurrence is the whole host.
/// The host must start the string or follow `/` or `@`, and be followed by `/`, or by `:` in
/// scp-style `git@host:owner/repo` only (in a URL `host:` is a port). So `notgithub.com/a/b` and
/// `github.community/a/b` are not `github.com`.
fn path_after_host<'a>(s: &'a str, host: &str) -> Option<&'a str> {
    let bytes = s.as_bytes();
    let mut from = 0;
    while let Some(offset) = s.get(from..)?.find(host) {
        let start = from + offset;
        let end = start + host.len();
        let after_userinfo = start > 0 && bytes[start - 1] == b'@';
        let starts_host = start == 0 || after_userinfo || bytes[start - 1] == b'/';
        if starts_host {
            let after = &s[end..];
            let path = after
                .strip_prefix('/')
                .or_else(|| after.strip_prefix(':').filter(|_| after_userinfo));
            if let Some(path) = path {
                return Some(path);
            }
        }
        from = end;
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssueRef {
    pub number: i64,
    /// "closes" if a closing keyword preceded the reference, else "mentions".
    pub relation: String,
}

/// GitHub's closing keywords (case-insensitive). A `#N` right after one is a closing reference.
const CLOSING_KEYWORDS: [&str; 9] = [
    "close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved",
];

/// Parse issue references (`#N`) out of text. Same-repo numeric references only. Each number is
/// reported once; a close wins over a mention.
pub fn parse_issue_references(text: &str) -> Vec<IssueRef> {
    let bytes = text.as_bytes();
    // Keep first-seen order while deduping and upgrading mentions to closes.
    let mut order: Vec<i64> = Vec::new();
    let mut relation_of: std::collections::HashMap<i64, String> = std::collections::HashMap::new();

    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start {
                if let Ok(number) = text[start..j].parse::<i64>() {
                    let closes = preceding_word_is_closing(&text[..i]);
                    let relation = if closes { "closes" } else { "mentions" };
                    match relation_of.get(&number) {
                        None => {
                            order.push(number);
                            relation_of.insert(number, relation.to_string());
                        }
                        Some(existing) if existing == "mentions" && closes => {
                            relation_of.insert(number, "closes".to_string());
                        }
                        Some(_) => {}
                    }
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    order
        .into_iter()
        .map(|number| IssueRef {
            relation: relation_of[&number].clone(),
            number,
        })
        .collect()
}

/// Parse Jira-style issue keys (`PROJ-123`) out of text: 2+ uppercase letters/digits starting
/// with a letter, a hyphen, then digits, at a token boundary. Liberal: the linker keeps only
/// keys matching a synced Jira issue. Deduped, first-seen order.
pub fn parse_issue_keys(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut i = 0;
    while i < n {
        let prev_boundary =
            i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if prev_boundary && bytes[i].is_ascii_uppercase() {
            let start = i;
            let mut j = i + 1;
            while j < n && (bytes[j].is_ascii_uppercase() || bytes[j].is_ascii_digit()) {
                j += 1;
            }
            // project part needs 2+ chars, then '-', then digits
            if j - start >= 2 && j < n && bytes[j] == b'-' {
                let dstart = j + 1;
                let mut k = dstart;
                while k < n && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                let after_boundary =
                    k >= n || !(bytes[k].is_ascii_alphanumeric() || bytes[k] == b'_');
                if k > dstart && after_boundary {
                    let key = &text[start..k];
                    if seen.insert(key) {
                        out.push(key.to_string());
                    }
                    i = k;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// True if the last whitespace-delimited word before `prefix` ends is a closing keyword.
/// Trailing punctuation (e.g. "fixes:") is stripped.
fn preceding_word_is_closing(prefix: &str) -> bool {
    match prefix.split_whitespace().next_back() {
        Some(word) => {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>()
                .to_lowercase();
            CLOSING_KEYWORDS.contains(&cleaned.as_str())
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(deps: &'a [Dependency], name: &str) -> &'a Dependency {
        deps.iter().find(|d| d.name == name).expect("dep present")
    }

    #[test]
    fn parses_jira_issue_keys() {
        assert_eq!(
            parse_issue_keys("Fix JENKINS-68355 and PROJ-7 (see PROJ-7 again)"),
            vec!["JENKINS-68355".to_string(), "PROJ-7".to_string()]
        );
        // Boundaries: not mid-identifier, digits must follow the hyphen, no trailing alnum.
        assert!(parse_issue_keys("xPROJ-1 PROJ-1x A-1 lower-7").is_empty());
        // A branch name plus title.
        assert_eq!(
            parse_issue_keys("JENKINS-12345-add-retry"),
            vec!["JENKINS-12345".to_string()]
        );
    }

    #[test]
    fn repo_on_host_resolves_go_module_paths() {
        // A go module path is the repo path; submodule and major-version suffixes drop to
        // owner/repo.
        assert_eq!(
            repo_on_host("go", "github.com/acme/widget", Some("v1.2.3"), "github.com"),
            Some("acme/widget".to_string())
        );
        assert_eq!(
            repo_on_host("go", "github.com/acme/widget/v2", None, "github.com"),
            Some("acme/widget".to_string())
        );
        assert_eq!(
            repo_on_host("go", "github.com/acme/widget/sub/pkg", None, "github.com"),
            Some("acme/widget".to_string())
        );
        // A module on another host (e.g. golang.org/x/...) does not resolve.
        assert_eq!(
            repo_on_host("go", "golang.org/x/sync", None, "github.com"),
            None
        );
    }

    #[test]
    fn repo_on_host_resolves_npm_and_pip_git_requirements() {
        // npm `github:` shorthand and git+https URLs (with .git and a #ref) live in version_req.
        assert_eq!(
            repo_on_host("npm", "widget", Some("github:acme/widget"), "github.com"),
            Some("acme/widget".to_string())
        );
        assert_eq!(
            repo_on_host(
                "npm",
                "widget",
                Some("git+https://github.com/acme/widget.git#v1"),
                "github.com"
            ),
            Some("acme/widget".to_string())
        );
        // pip `name @ url` keeps the URL in version_req.
        assert_eq!(
            repo_on_host(
                "pip",
                "widget",
                Some("@ https://github.com/acme/widget"),
                "github.com"
            ),
            Some("acme/widget".to_string())
        );
        // scp-style git@github.com:owner/repo.
        assert_eq!(
            repo_on_host(
                "npm",
                "widget",
                Some("git@github.com:acme/widget.git"),
                "github.com"
            ),
            Some("acme/widget".to_string())
        );
    }

    #[test]
    fn repo_on_host_returns_none_for_registry_named_deps() {
        // A crates.io / npm-registry dep named by package only encodes no repo.
        assert_eq!(
            repo_on_host("cargo", "serde", Some("1.0"), "github.com"),
            None
        );
        assert_eq!(
            repo_on_host("npm", "react", Some("^18.0.0"), "github.com"),
            None
        );
        assert_eq!(
            repo_on_host("pip", "requests", Some("==2.31"), "github.com"),
            None
        );
        assert_eq!(repo_on_host("npm", "react", None, "github.com"), None);
    }

    #[test]
    fn repo_on_host_resolves_a_self_hosted_forge_host() {
        // The same shapes against a Gitea/Forgejo host.
        assert_eq!(
            repo_on_host(
                "go",
                "gitea.example.com/acme/widget/v2",
                Some("v2.1.0"),
                "gitea.example.com"
            ),
            Some("acme/widget".to_string())
        );
        assert_eq!(
            repo_on_host(
                "npm",
                "widget",
                Some("git+https://gitea.example.com/acme/widget.git#v1"),
                "gitea.example.com"
            ),
            Some("acme/widget".to_string())
        );
        assert_eq!(
            repo_on_host(
                "pip",
                "widget",
                Some("@ git@gitea.example.com:acme/widget.git"),
                "gitea.example.com"
            ),
            Some("acme/widget".to_string())
        );
        // A github.com dependency does not resolve against the Gitea host, and the reverse.
        assert_eq!(
            repo_on_host("go", "github.com/acme/widget", None, "gitea.example.com"),
            None
        );
        assert_eq!(
            repo_on_host("go", "gitea.example.com/acme/widget", None, "github.com"),
            None
        );
    }

    #[test]
    fn github_shorthand_is_github_only() {
        // npm's `github:` shorthand names GitHub whatever host we resolve against.
        assert_eq!(
            repo_on_host(
                "npm",
                "widget",
                Some("github:acme/widget"),
                "gitea.example.com"
            ),
            None
        );
    }

    #[test]
    fn repo_on_host_needs_a_real_host_boundary() {
        // A host that merely contains the one we look for is a different host: self-hosted names
        // are often prefixes/suffixes of each other.
        assert_eq!(
            repo_on_host("go", "notgithub.com/acme/widget", None, "github.com"),
            None
        );
        assert_eq!(
            repo_on_host("go", "github.community/acme/widget", None, "github.com"),
            None
        );
        assert_eq!(
            repo_on_host("go", "git.acme.io/a/b", None, "notgit.acme.io"),
            None
        );
        // An empty host matches nothing rather than everything.
        assert_eq!(repo_on_host("go", "github.com/acme/widget", None, ""), None);
    }

    #[test]
    fn cargo_toml_extracts_all_three_tables_with_kinds() {
        let text = r#"
            [package]
            name = "demo"

            [dependencies]
            serde = "1"
            tokio = { version = "1", features = ["full"] }
            local = { path = "../local" }

            [dev-dependencies]
            wiremock = "0.6"

            [build-dependencies]
            cc = "1.0"
        "#;
        let deps = parse_cargo_toml(text).unwrap();
        assert_eq!(deps.len(), 5);
        assert_eq!(find(&deps, "serde").version_req.as_deref(), Some("1"));
        assert_eq!(find(&deps, "tokio").version_req.as_deref(), Some("1"));
        assert_eq!(find(&deps, "local").version_req, None);
        assert_eq!(find(&deps, "serde").kind, "normal");
        assert_eq!(find(&deps, "wiremock").kind, "dev");
        assert_eq!(find(&deps, "cc").kind, "build");
        assert!(deps.iter().all(|d| d.ecosystem == "cargo"));
    }

    #[test]
    fn cargo_toml_without_dependencies_is_empty_not_error() {
        let deps = parse_cargo_toml("[package]\nname = \"x\"\n").unwrap();
        assert!(deps.is_empty());
    }

    #[test]
    fn package_json_extracts_deps_and_dev_deps() {
        let text = r#"
            {
              "name": "demo",
              "dependencies": { "react": "^18.2.0", "zod": "3.23.8" },
              "devDependencies": { "vite": "^5.0.0" }
            }
        "#;
        let deps = parse_package_json(text).unwrap();
        assert_eq!(deps.len(), 3);
        assert_eq!(find(&deps, "react").version_req.as_deref(), Some("^18.2.0"));
        assert_eq!(find(&deps, "react").kind, "normal");
        assert_eq!(find(&deps, "vite").kind, "dev");
        assert!(deps.iter().all(|d| d.ecosystem == "npm"));
    }

    #[test]
    fn parse_manifest_dispatches_by_name_and_rejects_unknown() {
        assert_eq!(
            parse_manifest("Cargo.toml", "[dependencies]\nserde = \"1\"\n")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            parse_manifest("package.json", "{\"dependencies\":{\"a\":\"1\"}}")
                .unwrap()
                .len(),
            1
        );
        assert!(matches!(
            parse_manifest("Gemfile", ""),
            Err(GraphError::UnknownManifest(_))
        ));
    }

    #[test]
    fn malformed_cargo_toml_errors() {
        assert!(parse_cargo_toml("this is not toml = = =").is_err());
    }

    #[test]
    fn requirements_txt_parses_names_and_skips_noise() {
        let text = "# deps\nrequests==2.31.0\nflask>=1.0,<2\n\n-r other.txt\nplain-pkg\nnumpy  # pinned later\n";
        let deps = parse_requirements_txt(text);
        assert_eq!(deps.len(), 4); // requests, flask, plain-pkg, numpy
        assert!(deps.iter().all(|d| d.ecosystem == "pip"));
        assert_eq!(
            find(&deps, "requests").version_req.as_deref(),
            Some("==2.31.0")
        );
        assert_eq!(find(&deps, "plain-pkg").version_req, None);
        // The -r option line and the comment line are skipped.
        assert!(!deps
            .iter()
            .any(|d| d.name.starts_with('-') || d.name == "deps"));
    }

    #[test]
    fn pyproject_parses_project_and_optional_deps() {
        let text = r#"
            [project]
            name = "demo"
            dependencies = ["requests>=2", "click"]

            [project.optional-dependencies]
            test = ["pytest>=7", "coverage"]
        "#;
        let deps = parse_pyproject_toml(text).unwrap();
        assert_eq!(deps.len(), 4);
        assert_eq!(find(&deps, "requests").kind, "normal");
        assert_eq!(find(&deps, "pytest").kind, "dev");
        assert!(deps.iter().all(|d| d.ecosystem == "pip"));
    }

    #[test]
    fn go_mod_parses_block_and_indirect() {
        let text = "module example.com/demo\n\ngo 1.22\n\nrequire github.com/pkg/errors v0.9.1\n\nrequire (\n\tgolang.org/x/sync v0.7.0\n\tgithub.com/x/y v1.2.3 // indirect\n)\n";
        let deps = parse_go_mod(text);
        assert_eq!(deps.len(), 3);
        assert!(deps.iter().all(|d| d.ecosystem == "go"));
        assert_eq!(
            find(&deps, "github.com/pkg/errors").version_req.as_deref(),
            Some("v0.9.1")
        );
        assert_eq!(find(&deps, "golang.org/x/sync").kind, "normal");
        assert_eq!(find(&deps, "github.com/x/y").kind, "indirect");
    }

    #[test]
    fn parse_manifest_dispatches_the_new_ecosystems() {
        assert_eq!(
            parse_manifest("requirements.txt", "requests==2.0\n")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(parse_manifest("go.mod", "require m v1\n").unwrap().len(), 1);
        assert_eq!(
            parse_manifest("pyproject.toml", "[project]\ndependencies=[\"a\"]\n")
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn references_distinguish_closing_from_mention() {
        let refs = parse_issue_references("Fixes #10 and also see #11 for context.");
        assert_eq!(refs.len(), 2);
        let r10 = refs.iter().find(|r| r.number == 10).unwrap();
        assert_eq!(r10.relation, "closes");
        let r11 = refs.iter().find(|r| r.number == 11).unwrap();
        assert_eq!(r11.relation, "mentions");
    }

    #[test]
    fn references_dedup_and_upgrade_mention_to_close() {
        // #5 appears as a mention then as a close; the close wins, reported once.
        let refs = parse_issue_references("see #5 ... later this resolves #5");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].number, 5);
        assert_eq!(refs[0].relation, "closes");
    }

    #[test]
    fn references_handle_punctuation_and_keyword_case() {
        let refs = parse_issue_references("CLOSES: #42, plus (#43).");
        let r42 = refs.iter().find(|r| r.number == 42).unwrap();
        assert_eq!(r42.relation, "closes");
        let r43 = refs.iter().find(|r| r.number == 43).unwrap();
        assert_eq!(r43.relation, "mentions");
    }

    #[test]
    fn no_references_yields_empty() {
        assert!(parse_issue_references("a markdown # heading and no refs").is_empty());
    }
}
