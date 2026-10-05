//! Behavioral code analysis: deterministic, pure functions over a repo's file-change history.
//! No network, no LLM. Covers hotspots, churn, ownership risk, coupling and a health score.
//! Inputs are PR-granular (`pr_files` joined to `pull_requests`). Source text is not stored, so
//! churn (lines changed) stands in for size in the hotspot score.

use serde::{Deserialize, Serialize};

/// Per-file change activity, PR-granular. Input to hotspot analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileActivity {
    pub path: String,
    /// Number of PRs that touched this file.
    pub changes: i64,
    pub additions: i64,
    pub deletions: i64,
    /// Distinct PR-author logins that touched it.
    pub authors: i64,
}

/// A hotspot: a file that changes often and churns a lot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hotspot {
    pub path: String,
    pub changes: i64,
    /// Total lines changed (additions + deletions).
    pub churn: i64,
    pub authors: i64,
    /// Hotspot score = change frequency x churn. Higher = hotter.
    pub score: f64,
}

/// True for generated or vendored paths (lockfiles etc.), which would show up as false hotspots.
pub fn is_generated(path: &str) -> bool {
    let p = path.to_ascii_lowercase();
    const DIR_MARKERS: [&str; 8] = [
        "node_modules/",
        "vendor/",
        "dist/",
        "build/",
        "target/",
        ".min.",
        "generated/",
        "third_party/",
    ];
    if DIR_MARKERS.iter().any(|m| p.contains(m)) {
        return true;
    }
    const LOCKFILES: [&str; 7] = [
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "cargo.lock",
        "go.sum",
        "poetry.lock",
        "composer.lock",
    ];
    let file = p.rsplit('/').next().unwrap_or(&p);
    LOCKFILES.contains(&file)
}

/// Rank files by change frequency x churn, excluding generated paths and files that never churned.
/// Returns the top `limit`, ordered by score desc, then path.
pub fn hotspots(files: &[FileActivity], limit: usize) -> Vec<Hotspot> {
    let mut spots: Vec<Hotspot> = files
        .iter()
        .filter(|f| !is_generated(&f.path))
        .map(|f| {
            let churn = f.additions + f.deletions;
            Hotspot {
                path: f.path.clone(),
                changes: f.changes,
                churn,
                authors: f.authors,
                score: f.changes as f64 * churn as f64,
            }
        })
        .filter(|h| h.score > 0.0)
        .collect();
    spots.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    spots.truncate(limit);
    spots
}

// ---- ownership risk ---------------------------------------------
// A person may be named as a module's sole owner, but is never ranked or scored against peers.

/// One (file, author) change count, PR-granular. Input to ownership analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileAuthorship {
    pub path: String,
    pub author: String,
    /// Number of PRs by this author touching this file.
    pub changes: i64,
}

/// A module's ownership risk: change activity, authorship concentration, and bus factor (authors
/// who would have to leave before most of the module is orphaned).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleOwnership {
    /// The module (directory) path, or "(root)" for top-level files.
    pub module: String,
    pub changes: i64,
    pub authors: i64,
    /// The author with the largest share of this module's changes.
    pub top_author: String,
    /// The top author's share of changes, `0.0..=1.0`.
    pub top_share: f64,
    /// Minimum authors whose combined changes exceed 50%. Low means key-person risk.
    pub bus_factor: i64,
}

/// The module (directory) a path belongs to: everything before the last `/`, or "(root)".
fn module_of(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[..i].to_string(),
        None => "(root)".to_string(),
    }
}

/// Authorship concentration for changes keyed by author: dominant author, its share, bus factor
/// (minimum authors whose combined changes exceed 50%) and author count. Empty input gives no
/// owner, share 0, bus factor 0.
fn concentration(by_author: &std::collections::BTreeMap<String, i64>) -> (String, f64, i64, i64) {
    let total: i64 = by_author.values().sum();
    if total == 0 {
        return (String::new(), 0.0, 0, by_author.len() as i64);
    }
    // Authors sorted by descending change count (ties by name for determinism).
    let mut ranked: Vec<(&String, &i64)> = by_author.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let (top_author, top_changes) = ranked[0];
    let mut acc = 0i64;
    let mut bus_factor = 0i64;
    for (_, c) in &ranked {
        acc += **c;
        bus_factor += 1;
        if acc * 2 > total {
            break;
        }
    }
    (
        top_author.to_string(),
        *top_changes as f64 / total as f64,
        bus_factor,
        by_author.len() as i64,
    )
}

/// Per-module ownership concentration and bus factor. Excludes generated paths and modules below
/// `min_changes`; ranks lowest bus factor first, then most active. Heuristic (PR-granular).
pub fn module_ownership(files: &[FileAuthorship], min_changes: i64) -> Vec<ModuleOwnership> {
    use std::collections::BTreeMap;
    // module -> (author -> changes)
    let mut mods: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    for f in files.iter().filter(|f| !is_generated(&f.path)) {
        *mods
            .entry(module_of(&f.path))
            .or_default()
            .entry(f.author.clone())
            .or_default() += f.changes;
    }
    let mut out: Vec<ModuleOwnership> = mods
        .into_iter()
        .filter_map(|(module, by_author)| {
            let total: i64 = by_author.values().sum();
            if total < min_changes {
                return None;
            }
            let (top_author, top_share, bus_factor, authors) = concentration(&by_author);
            Some(ModuleOwnership {
                module,
                changes: total,
                authors,
                top_author,
                top_share,
                bus_factor,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        a.bus_factor
            .cmp(&b.bus_factor)
            .then_with(|| b.changes.cmp(&a.changes))
            .then_with(|| a.module.cmp(&b.module))
    });
    out
}

// ---- code risk fusion -------------------------------------------
// One per-file score fusing hotspot, ownership concentration and merged-unreviewed rate. Every row
// keeps its component numbers and a plain-language "why", so the score is a ranking aid.

/// Per-file merged-PR review coverage: `merged_touches` merged PRs touched the file, `unreviewed`
/// of them had no review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileReview {
    pub path: String,
    pub merged_touches: i64,
    pub unreviewed: i64,
}

/// A file's fused code risk: the hotspot, ownership, and review components, a composite score
/// for ranking, and the plain-language reasons behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeRisk {
    pub path: String,
    pub changes: i64,
    pub churn: i64,
    pub authors: i64,
    pub top_author: String,
    pub top_share: f64,
    pub bus_factor: i64,
    pub merged_touches: i64,
    pub unreviewed: i64,
    /// Share of merged touches that landed with no review, `0.0..=1.0`.
    pub review_gap: f64,
    /// Composite = hotspot (changes x churn) x (1 + top_share) x (1 + review_gap).
    pub score: f64,
    /// Why this file is risky: hotspot, plus single owner and unreviewed merges when they apply.
    pub reasons: Vec<String>,
}

/// Fuse per-file hotspot, ownership and review signals into a ranked code-risk list, riskiest
/// first, up to `limit`. Generated paths and files with no churn are excluded. `authorship` is per
/// (file, author); `reviews` is per file.
pub fn code_risk(
    activity: &[FileActivity],
    authorship: &[FileAuthorship],
    reviews: &[FileReview],
    limit: usize,
) -> Vec<CodeRisk> {
    use std::collections::BTreeMap;
    // Per-file author -> changes, for concentration.
    let mut by_file: BTreeMap<&str, BTreeMap<String, i64>> = BTreeMap::new();
    for a in authorship.iter().filter(|a| !is_generated(&a.path)) {
        *by_file
            .entry(a.path.as_str())
            .or_default()
            .entry(a.author.clone())
            .or_default() += a.changes;
    }
    let review_of: std::collections::HashMap<&str, &FileReview> =
        reviews.iter().map(|r| (r.path.as_str(), r)).collect();

    let mut out: Vec<CodeRisk> = activity
        .iter()
        .filter(|f| !is_generated(&f.path))
        .filter_map(|f| {
            let churn = f.additions + f.deletions;
            let hotspot = f.changes as f64 * churn as f64;
            if hotspot <= 0.0 {
                return None;
            }
            let (top_author, top_share, bus_factor, _) = by_file
                .get(f.path.as_str())
                .map(concentration)
                .unwrap_or_default();
            let (merged_touches, unreviewed) = review_of
                .get(f.path.as_str())
                .map(|r| (r.merged_touches, r.unreviewed))
                .unwrap_or((0, 0));
            let review_gap = if merged_touches > 0 {
                unreviewed as f64 / merged_touches as f64
            } else {
                0.0
            };
            let score = hotspot * (1.0 + top_share) * (1.0 + review_gap);

            let mut reasons = vec![format!("{} changes, {} lines churned", f.changes, churn)];
            if bus_factor == 1 && !top_author.is_empty() {
                reasons.push(format!(
                    "single owner {} ({:.0}%)",
                    top_author,
                    top_share * 100.0
                ));
            }
            if review_gap > 0.0 {
                reasons.push(format!(
                    "{:.0}% of {} merged change(s) landed unreviewed",
                    review_gap * 100.0,
                    merged_touches
                ));
            }
            Some(CodeRisk {
                path: f.path.clone(),
                changes: f.changes,
                churn,
                authors: f.authors,
                top_author,
                top_share,
                bus_factor,
                merged_touches,
                unreviewed,
                review_gap,
                score,
                reasons,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    out.truncate(limit);
    out
}

// ---- change/temporal coupling ---------------------------------------------
// Files that co-appear in the same merged PR more than by chance are likely coupled.

/// Co-occurrence counts from the store: two files, merged PRs containing both, and each file's own
/// merged-PR count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePairCo {
    pub path_a: String,
    pub path_b: String,
    /// Merged PRs where both files appeared.
    pub together: i64,
    /// Total merged PRs that touched path_a.
    pub prs_a: i64,
    /// Total merged PRs that touched path_b.
    pub prs_b: i64,
}

/// A change-coupling pair: two files that often appear in the same merged PR. Score is the Jaccard
/// index `together / (prs_a + prs_b - together)`; near 1.0 co-change, near 0.0 coincidental.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouplingPair {
    pub path_a: String,
    pub path_b: String,
    /// Merged PRs where both files changed together.
    pub together: i64,
    /// Jaccard coupling strength: `together / (prs_a + prs_b - together)`.
    pub coupling: f64,
}

/// Top change-coupling pairs: non-generated file pairs with at least `min_together` shared merged
/// PRs, ranked by Jaccard (ties: more co-occurrences, then path), up to `limit`.
pub fn coupling_pairs(raw: &[FilePairCo], min_together: i64, limit: usize) -> Vec<CouplingPair> {
    let mut out: Vec<CouplingPair> = raw
        .iter()
        .filter(|p| !is_generated(&p.path_a) && !is_generated(&p.path_b))
        .filter(|p| p.together >= min_together)
        .filter_map(|p| {
            let union = p.prs_a + p.prs_b - p.together;
            if union <= 0 {
                return None;
            }
            Some(CouplingPair {
                path_a: p.path_a.clone(),
                path_b: p.path_b.clone(),
                together: p.together,
                coupling: p.together as f64 / union as f64,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.coupling
            .partial_cmp(&a.coupling)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.together.cmp(&a.together))
            .then_with(|| a.path_a.cmp(&b.path_a))
            .then_with(|| a.path_b.cmp(&b.path_b))
    });
    out.truncate(limit);
    out
}

// ---- AST code-health score -------------------------------------
// Per-file complexity metrics from tree-sitter, mapped to a 1-10 health score (10 = simplest).
// Every component (LOC, function count, branch count) is kept on the output.

/// Per-file AST complexity metrics, stored in `file_health`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMetrics {
    /// Lines of code (including blank lines and comments). Needs no grammar.
    pub loc: u32,
    /// Whether the file was parsed. False without a tree-sitter grammar or on parse failure; the
    /// counts below are then meaningless, not zero.
    pub analyzed: bool,
    /// Number of function/method declarations found via tree-sitter. Only meaningful when
    /// `analyzed`.
    pub functions: u32,
    /// Number of decision points (if/while/for/match/switch arms) - a cyclomatic complexity
    /// proxy. Only meaningful when `analyzed`.
    pub branches: u32,
}

impl FileMetrics {
    /// LOC-only metrics for a file we could not parse, distinct from a real zero count.
    fn unanalyzed(loc: u32) -> Self {
        Self {
            loc,
            analyzed: false,
            functions: 0,
            branches: 0,
        }
    }
}

/// Compute per-file complexity metrics, using the chunker's grammar map
/// (`core_embed::tree_sitter_language`) keyed on a `source_language` label. No grammar or a failed
/// parse yields LOC-only metrics with `analyzed = false`.
pub fn compute_metrics(text: &str, lang: &str) -> FileMetrics {
    let loc = text.lines().count() as u32;
    let Some(language) = core_embed::tree_sitter_language(lang) else {
        return FileMetrics::unanalyzed(loc);
    };
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return FileMetrics::unanalyzed(loc);
    }
    let Some(tree) = parser.parse(text, None) else {
        return FileMetrics::unanalyzed(loc);
    };
    let (functions, branches) = walk_tree(
        tree.root_node(),
        function_node_kinds(lang),
        branch_node_kinds(lang),
    );
    FileMetrics {
        loc,
        analyzed: true,
        functions,
        branches,
    }
}

/// Derive a 1-10 health score from per-file metrics. Higher = simpler. `None` for a file that was
/// not parsed, since a score of 10 would read as "verified simple".
/// Start at 10 and deduct for branch density (`branches / functions`) and file size. A parsed file
/// with no detected function is scored as if it had one, which avoids dividing by zero and counts
/// the whole file as a single unit.
/// Thresholds, on the same float comparison the code makes:
/// - branch density at most 1: no penalty
/// - above 1, at most 3: 1 point
/// - above 3, at most 6: 2 points
/// - above 6, at most 10: 4 points
/// - above 10, at most 15: 6 points
/// - above 15: 8 points
/// - LOC 501-1000: 1 point
/// - LOC above 1000: 2 points
pub fn health_score(metrics: &FileMetrics) -> Option<u8> {
    if !metrics.analyzed {
        return None;
    }
    let effective_fns = metrics.functions.max(1) as f64;
    let avg_branches = metrics.branches as f64 / effective_fns;
    let complexity_penalty: i32 = if avg_branches <= 1.0 {
        0
    } else if avg_branches <= 3.0 {
        1
    } else if avg_branches <= 6.0 {
        2
    } else if avg_branches <= 10.0 {
        4
    } else if avg_branches <= 15.0 {
        6
    } else {
        8
    };
    let size_penalty: i32 = match metrics.loc {
        0..=500 => 0,
        501..=1000 => 1,
        _ => 2,
    };
    let raw = 10 - complexity_penalty - size_penalty;
    Some(raw.clamp(1, 10) as u8)
}

/// Walk the tree-sitter tree, returning `(functions, branches)` node counts.
/// Iterative on an explicit heap stack: this runs on a tokio worker (~2MB stack), a stack overflow
/// aborts the process, and a generated file within `MAX_CODE_FILE_BYTES` can nest thousands deep.
/// Only named nodes count. Some grammars expose a keyword as both a named node and an anonymous
/// token of the same type string (tree-sitter-ruby: `if`, `elsif`, `unless`, ...), which would
/// double count.
fn walk_tree(root: tree_sitter::Node, fn_kinds: &[&str], br_kinds: &[&str]) -> (u32, u32) {
    let mut functions = 0u32;
    let mut branches = 0u32;
    let mut cursor = root.walk();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.is_named() {
            let kind = node.kind();
            if fn_kinds.contains(&kind) {
                functions += 1;
            } else if br_kinds.contains(&kind) {
                branches += 1;
            }
        }
        stack.extend(node.children(&mut cursor));
    }
    (functions, branches)
}

/// Tree-sitter node kinds that represent function/method declarations per language.
fn function_node_kinds(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &["function_item"],
        "python" => &["function_definition"],
        "typescript" | "javascript" => &[
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
            "generator_function_declaration",
        ],
        "go" => &["function_declaration", "method_declaration", "func_literal"],
        "java" => &["method_declaration", "constructor_declaration"],
        "c" | "cpp" => &["function_definition"],
        "ruby" => &["method", "singleton_method"],
        "csharp" => &[
            "method_declaration",
            "constructor_declaration",
            "local_function_statement",
        ],
        "php" => &["function_definition", "method_declaration"],
        "shell" => &["function_definition"],
        _ => &[],
    }
}

/// Tree-sitter node kinds that represent decision/branch points per language.
fn branch_node_kinds(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &[
            "if_expression",
            "match_arm",
            "while_expression",
            "for_expression",
            "loop_expression",
        ],
        "python" => &[
            "if_statement",
            "elif_clause",
            "while_statement",
            "for_statement",
            "conditional_expression",
        ],
        "typescript" | "javascript" => &[
            "if_statement",
            "while_statement",
            "for_statement",
            "for_in_statement",
            "ternary_expression",
            "switch_case",
        ],
        // `expression_case` / `type_case` are the switch-arm kinds in tree-sitter-go.
        "go" => &[
            "if_statement",
            "for_statement",
            "expression_case",
            "type_case",
        ],
        "java" => &[
            "if_statement",
            "while_statement",
            "for_statement",
            "enhanced_for_statement",
            "switch_block_statement_group",
        ],
        "c" | "cpp" => &[
            "if_statement",
            "while_statement",
            "for_statement",
            "do_statement",
        ],
        // Block forms only. Modifier forms (`x if y`) are `if_modifier` etc. and are undercounted.
        "ruby" => &["if", "elsif", "unless", "while", "until", "for", "when"],
        "csharp" => &[
            "if_statement",
            "while_statement",
            "for_statement",
            "foreach_statement",
        ],
        "php" => &[
            "if_statement",
            "while_statement",
            "for_statement",
            "foreach_statement",
        ],
        "shell" => &[
            "if_statement",
            "while_statement",
            "for_statement",
            "c_style_for_statement",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fa(path: &str, changes: i64, add: i64, del: i64, authors: i64) -> FileActivity {
        FileActivity {
            path: path.into(),
            changes,
            additions: add,
            deletions: del,
            authors,
        }
    }

    #[test]
    fn generated_paths_are_excluded() {
        assert!(is_generated("ui/pnpm-lock.yaml"));
        assert!(is_generated("Cargo.lock"));
        assert!(is_generated("web/node_modules/x/index.js"));
        assert!(is_generated("dist/app.min.js"));
        assert!(!is_generated("src/core/engine.rs"));
    }

    #[test]
    fn hotspots_rank_by_frequency_times_churn() {
        let files = vec![
            fa("src/hot.rs", 10, 400, 100, 4),   // score 10 * 500 = 5000
            fa("src/warm.rs", 3, 200, 100, 2),   // 3 * 300 = 900
            fa("Cargo.lock", 50, 9999, 9999, 1), // excluded (generated)
            fa("src/cold.rs", 1, 0, 0, 1),       // churn 0 -> dropped
        ];
        let hs = hotspots(&files, 10);
        assert_eq!(hs.len(), 2);
        assert_eq!(hs[0].path, "src/hot.rs");
        assert_eq!(hs[0].churn, 500);
        assert_eq!(hs[0].score, 5000.0);
        assert_eq!(hs[1].path, "src/warm.rs");
    }

    #[test]
    fn limit_caps_the_result() {
        let files: Vec<FileActivity> = (0..20)
            .map(|i| fa(&format!("src/f{i}.rs"), i + 1, 10, 10, 1))
            .collect();
        assert_eq!(hotspots(&files, 5).len(), 5);
    }

    fn auth(path: &str, who: &str, changes: i64) -> FileAuthorship {
        FileAuthorship {
            path: path.into(),
            author: who.into(),
            changes,
        }
    }

    #[test]
    fn module_ownership_flags_single_maintainer() {
        let files = vec![
            // payments/: all alice -> bus factor 1, 100% concentration.
            auth("payments/api.rs", "alice", 8),
            auth("payments/db.rs", "alice", 4),
            // web/: shared across 3 -> higher bus factor, lower concentration.
            auth("web/a.rs", "bob", 5),
            auth("web/b.rs", "carol", 5),
            auth("web/c.rs", "dave", 4),
            // a generated path is excluded.
            auth("node_modules/x.js", "alice", 99),
            // a tiny module below min_changes is dropped.
            auth("scripts/once.sh", "eve", 1),
        ];
        let mods = module_ownership(&files, 3);
        let pay = mods.iter().find(|m| m.module == "payments").unwrap();
        assert_eq!(pay.bus_factor, 1);
        assert_eq!(pay.authors, 1);
        assert_eq!(pay.top_author, "alice");
        assert_eq!(pay.top_share, 1.0);
        let web = mods.iter().find(|m| m.module == "web").unwrap();
        assert_eq!(web.authors, 3);
        assert_eq!(web.bus_factor, 2); // need 2 of 3 to exceed 50%
        assert!(mods.iter().all(|m| m.module != "scripts")); // below min_changes
        assert!(mods.iter().all(|m| !m.module.contains("node_modules"))); // generated
        assert_eq!(mods[0].module, "payments");
    }

    #[test]
    fn module_of_uses_the_directory() {
        assert_eq!(module_of("src/core/engine.rs"), "src/core");
        assert_eq!(module_of("README.md"), "(root)");
    }

    #[test]
    fn code_risk_fuses_hotspot_ownership_and_review() {
        // Two equally-hot files (10 changes, 300 churn). auth.rs is single-owner, 60% unreviewed;
        // list.tsx is broadly owned and fully reviewed -> auth.rs is the riskier of the two.
        let activity = vec![
            fa("core/auth.rs", 10, 200, 100, 1),
            fa("ui/list.tsx", 10, 200, 100, 3),
            fa("Cargo.lock", 99, 9999, 9999, 1), // generated -> excluded
            fa("docs/note.md", 5, 0, 0, 1),      // zero churn -> excluded
        ];
        let authorship = vec![
            auth("core/auth.rs", "alice", 10),
            auth("ui/list.tsx", "alice", 4),
            auth("ui/list.tsx", "bob", 3),
            auth("ui/list.tsx", "carol", 3),
        ];
        let reviews = vec![
            FileReview {
                path: "core/auth.rs".into(),
                merged_touches: 5,
                unreviewed: 3,
            },
            FileReview {
                path: "ui/list.tsx".into(),
                merged_touches: 5,
                unreviewed: 0,
            },
        ];
        let risks = code_risk(&activity, &authorship, &reviews, 10);
        assert_eq!(risks.len(), 2, "generated + zero-churn files are excluded");
        // auth.rs leads: single owner + unreviewed amplify its score above list.tsx.
        assert_eq!(risks[0].path, "core/auth.rs");
        assert_eq!(risks[0].bus_factor, 1);
        assert_eq!(risks[0].top_author, "alice");
        assert!((risks[0].review_gap - 0.6).abs() < 1e-9);
        assert!(risks[0].reasons.iter().any(|r| r.contains("single owner")));
        assert!(risks[0].reasons.iter().any(|r| r.contains("unreviewed")));
        // list.tsx: broadly owned, fully reviewed -> only the hotspot reason, no review gap.
        let list = risks.iter().find(|r| r.path == "ui/list.tsx").unwrap();
        assert_eq!(list.review_gap, 0.0);
        assert_eq!(list.reasons.len(), 1);
        assert!(list.score < risks[0].score);
    }

    fn pair(a: &str, b: &str, together: i64, prs_a: i64, prs_b: i64) -> FilePairCo {
        FilePairCo {
            path_a: a.into(),
            path_b: b.into(),
            together,
            prs_a,
            prs_b,
        }
    }

    #[test]
    fn coupling_pairs_ranks_by_jaccard_excludes_generated() {
        let raw = vec![
            // auth.rs + db.rs: always co-change (Jaccard = 5 / (5+5-5) = 1.0).
            pair("core/auth.rs", "core/db.rs", 5, 5, 5),
            // api.rs + handler.rs: Jaccard = 3 / (10+4-3) = 3/11 ~ 0.27.
            pair("src/api.rs", "src/handler.rs", 3, 10, 4),
            // lockfile: excluded (generated).
            pair("Cargo.lock", "core/db.rs", 5, 5, 5),
            // below min_together=2: dropped.
            pair("src/rare_a.rs", "src/rare_b.rs", 1, 1, 1),
        ];
        let pairs = coupling_pairs(&raw, 2, 10);
        // Two results: auth+db and api+handler. Lockfile excluded; rare pair dropped.
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].path_a, "core/auth.rs");
        assert_eq!(pairs[0].path_b, "core/db.rs");
        assert!((pairs[0].coupling - 1.0).abs() < 1e-9);
        assert_eq!(pairs[1].path_a, "src/api.rs");
        assert!((pairs[1].coupling - 3.0 / 11.0).abs() < 1e-9);
        let capped = coupling_pairs(&raw, 2, 1);
        assert_eq!(capped.len(), 1);
        assert_eq!(capped[0].path_a, "core/auth.rs");
    }

    // ---- AST metrics + health score --------------------------------

    /// Metrics for a file that was parsed, so `health_score` returns a number.
    fn parsed(loc: u32, functions: u32, branches: u32) -> FileMetrics {
        FileMetrics {
            loc,
            analyzed: true,
            functions,
            branches,
        }
    }

    #[test]
    fn health_score_simple_file_scores_high() {
        // Parsed, one function, no branches: nothing to deduct.
        assert_eq!(health_score(&parsed(50, 1, 0)), Some(10));
    }

    #[test]
    fn health_score_complex_file_scores_low() {
        // 30 branches across 2 functions -> density 15, penalty 6; plus >1000 LOC penalty 2.
        assert_eq!(health_score(&parsed(1200, 2, 30)), Some(2));
    }

    #[test]
    fn health_score_clamped_to_1() {
        assert_eq!(health_score(&parsed(2000, 1, 100)), Some(1));
    }

    #[test]
    fn health_score_pins_the_published_band_edges() {
        // Branch density: each band is "at most", so the edge value stays in the lower band and
        // the next representable step above it falls through to the next one.
        assert_eq!(
            health_score(&parsed(10, 1, 1)),
            Some(10),
            "density 1.0: no penalty"
        );
        assert_eq!(
            health_score(&parsed(10, 2, 3)),
            Some(9),
            "density 1.5: 1 point"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 3)),
            Some(9),
            "density 3.0: 1 point"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 4)),
            Some(8),
            "density 4.0: 2 points"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 6)),
            Some(8),
            "density 6.0: 2 points"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 7)),
            Some(6),
            "density 7.0: 4 points"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 10)),
            Some(6),
            "density 10.0: 4 points"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 11)),
            Some(4),
            "density 11.0: 6 points"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 15)),
            Some(4),
            "density 15.0: 6 points"
        );
        assert_eq!(
            health_score(&parsed(10, 1, 16)),
            Some(2),
            "density 16.0: 8 points"
        );
        // Size: 500 is free, 501 costs a point, 1000 still one, 1001 costs two.
        assert_eq!(health_score(&parsed(500, 1, 0)), Some(10));
        assert_eq!(health_score(&parsed(501, 1, 0)), Some(9));
        assert_eq!(health_score(&parsed(1000, 1, 0)), Some(9));
        assert_eq!(health_score(&parsed(1001, 1, 0)), Some(8));
    }

    #[test]
    fn health_score_treats_a_function_less_parsed_file_as_one_function() {
        // The documented max(1) fudge: 4 top-level branches and no function is density 4.0, not a
        // division by zero and not a free pass.
        assert_eq!(health_score(&parsed(10, 0, 4)), Some(8));
        assert_eq!(health_score(&parsed(10, 0, 0)), Some(10));
    }

    #[test]
    fn compute_metrics_rust_counts_functions_and_branches() {
        let src = r#"
fn simple(x: i32) -> i32 {
    if x > 0 { x } else { -x }
}

fn loopy(n: usize) {
    for _ in 0..n {
        if n > 10 {
            while false {}
        }
    }
}
"#;
        let m = compute_metrics(src, "rust");
        assert!(m.analyzed);
        assert_eq!(m.functions, 2, "two fn items");
        // 2 if_expression + 1 for_expression + 1 while_expression.
        assert_eq!(m.branches, 4);
        assert_eq!(m.loc, 12);
    }

    #[test]
    fn compute_metrics_python_counts_functions_and_branches() {
        let src = r#"
def greet(name):
    if name:
        return f"Hello, {name}"
    else:
        return "Hello"

def loop_example(items):
    for item in items:
        if item > 0:
            print(item)
"#;
        let m = compute_metrics(src, "python");
        assert!(m.analyzed);
        assert_eq!(m.functions, 2, "two function_definition nodes");
        // 2 if_statement + 1 for_statement. The `else:` is a clause of its `if`, not a branch.
        assert_eq!(m.branches, 3);
    }

    #[test]
    fn compute_metrics_ruby_counts_each_conditional_once() {
        // tree-sitter-ruby exposes these keywords as both named and anonymous nodes; pins the
        // named-only walk.
        let src = r#"
def classify(n)
  if n > 10
    :big
  elsif n > 5
    :medium
  else
    :small
  end
  while n > 0
    n -= 1
  end
  case n
  when 0 then :zero
  when 1 then :one
  end
  puts n unless n.zero?
end
"#;
        let m = compute_metrics(src, "ruby");
        assert!(m.analyzed);
        assert_eq!(m.functions, 1, "one method");
        // if + elsif + while + 2 when = 5. The trailing `unless` modifier parses as
        // `unless_modifier`, which is not in the kind table (known undercount).
        assert_eq!(m.branches, 5);
    }

    #[test]
    fn compute_metrics_go_counts_switch_cases() {
        // Go switch arms are `expression_case` / `type_case`.
        let src = r#"
package main

func dispatch(v interface{}) string {
	switch x := v.(type) {
	case int:
		return "int"
	case string:
		return "string"
	}
	switch n := 1; n {
	case 1:
		return "one"
	case 2:
		return "two"
	default:
		return "many"
	}
}

func loop(n int) {
	for i := 0; i < n; i++ {
		if i > 2 {
			return
		}
	}
}
"#;
        let m = compute_metrics(src, "go");
        assert!(m.analyzed);
        assert_eq!(m.functions, 2, "two function declarations");
        // 2 type_case + 2 expression_case + 1 for_statement + 1 if_statement. `default` is a
        // `default_case`, which is not a decision point, so it is not counted.
        assert_eq!(m.branches, 6);
    }

    #[test]
    fn compute_metrics_language_without_a_grammar_is_not_analyzed_and_has_no_score() {
        // kotlin/swift/scala/sql have no tree-sitter grammar; without `analyzed` they score 10.
        for lang in ["kotlin", "swift", "scala", "sql", "cobol"] {
            let m = compute_metrics("line 1\nline 2\nline 3\n", lang);
            assert_eq!(m.loc, 3, "{lang}: LOC is still real");
            assert!(!m.analyzed, "{lang}: no grammar, so not analyzed");
            assert_eq!(
                health_score(&m),
                None,
                "{lang}: no score for an unparsed file"
            );
        }
    }

    #[test]
    fn compute_metrics_walks_a_deeply_nested_file_without_overflowing_the_stack() {
        // A minified bundle within the byte cap can nest deeper than a tokio worker's ~2MB stack;
        // a recursive walk would abort the test binary.
        const DEPTH: usize = 5_000;
        let mut src = String::from("function f() {\n");
        for _ in 0..DEPTH {
            src.push_str("if (a) {");
        }
        src.push_str("b();");
        for _ in 0..DEPTH {
            src.push('}');
        }
        src.push_str("\n}\n");
        let m = compute_metrics(&src, "javascript");
        assert!(m.analyzed);
        assert_eq!(m.functions, 1);
        assert_eq!(m.branches, DEPTH as u32);
    }
}
