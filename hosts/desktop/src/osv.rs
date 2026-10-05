//! OSV advisory lookups for dependency vulnerability scanning. Queries osv.dev's batch API for a
//! board's dependency packages and returns those with known advisories. Egress only when the
//! user runs a scan on a scan-enabled board.
//! Manifests carry version ranges, not resolved versions, so the query is package-level: "this
//! package has known advisories; verify your pinned version".

use serde::Deserialize;

/// A dependency package that has known advisories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageAdvisory {
    pub ecosystem: String,
    pub name: String,
    pub advisories: Vec<String>,
}

/// Map orgonzola's ecosystem name to the OSV ecosystem string; `None` if OSV does not cover it.
fn osv_ecosystem(eco: &str) -> Option<&'static str> {
    match eco {
        "cargo" => Some("crates.io"),
        "npm" => Some("npm"),
        "pip" => Some("PyPI"),
        "go" => Some("Go"),
        _ => None,
    }
}

#[derive(Deserialize)]
struct OsvBatchResponse {
    results: Vec<OsvResult>,
}

#[derive(Deserialize, Default)]
struct OsvResult {
    #[serde(default)]
    vulns: Vec<OsvVuln>,
}

#[derive(Deserialize)]
struct OsvVuln {
    id: String,
}

/// Parse an OSV `querybatch` response, aligning results to `queried` (the order sent), and
/// return only packages with advisories.
fn parse_osv_batch(
    queried: &[(String, String)],
    body: &str,
) -> Result<Vec<PackageAdvisory>, String> {
    let resp: OsvBatchResponse =
        serde_json::from_str(body).map_err(|e| format!("OSV response not understood: {e}"))?;
    let mut out = Vec::new();
    for ((ecosystem, name), result) in queried.iter().zip(resp.results.iter()) {
        if result.vulns.is_empty() {
            continue;
        }
        out.push(PackageAdvisory {
            ecosystem: ecosystem.clone(),
            name: name.clone(),
            advisories: result.vulns.iter().map(|v| v.id.clone()).collect(),
        });
    }
    Ok(out)
}

/// The OSV batch endpoint (tests pass their own URL).
pub const OSV_BATCH_URL: &str = "https://api.osv.dev/v1/querybatch";

/// Query OSV for advisories on `packages` (each `(ecosystem, name)`). Packages in OSV-unknown
/// ecosystems are skipped and duplicates removed. Network egress.
pub async fn query_osv(
    base_url: &str,
    packages: &[(String, String)],
) -> Result<Vec<PackageAdvisory>, String> {
    // Dedup and keep OSV-known ecosystems in a stable order. `queried` keeps the original
    // ecosystem label (advisories are reported under it); the OSV-mapped name is used only for
    // the query.
    let mut seen = std::collections::BTreeSet::new();
    let queried: Vec<(String, String)> = packages
        .iter()
        .filter(|(eco, _)| osv_ecosystem(eco).is_some())
        .filter(|p| seen.insert((p.0.clone(), p.1.clone())))
        .cloned()
        .collect();
    if queried.is_empty() {
        return Ok(Vec::new());
    }
    let queries: Vec<serde_json::Value> = queried
        .iter()
        .map(|(eco, name)| {
            // `queried` holds only OSV-known ecosystems, so the fallback only avoids a panic.
            serde_json::json!({
                "package": { "ecosystem": osv_ecosystem(eco).unwrap_or(eco.as_str()), "name": name }
            })
        })
        .collect();
    let client = reqwest::Client::builder()
        .user_agent("orgonzola")
        .build()
        .map_err(|e| e.to_string())?;
    let body = client
        .post(base_url)
        .json(&serde_json::json!({ "queries": queries }))
        .send()
        .await
        .map_err(|e| format!("OSV request failed: {e}"))?
        .text()
        .await
        .map_err(|e| format!("OSV request failed: {e}"))?;
    parse_osv_batch(&queried, &body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_aligns_results_and_keeps_only_hits() {
        let queried = vec![
            ("cargo".to_string(), "serde".to_string()),
            ("npm".to_string(), "left-pad".to_string()),
            ("cargo".to_string(), "tokio".to_string()),
        ];
        // serde clean, left-pad has 2 advisories, tokio has 1.
        let body = r#"{"results":[
            {},
            {"vulns":[{"id":"GHSA-aaaa"},{"id":"CVE-2020-1"}]},
            {"vulns":[{"id":"RUSTSEC-2021-0001"}]}
        ]}"#;
        let hits = parse_osv_batch(&queried, body).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].name, "left-pad");
        assert_eq!(hits[0].advisories.len(), 2);
        assert_eq!(hits[1].name, "tokio");
        assert_eq!(hits[1].advisories, vec!["RUSTSEC-2021-0001"]);
    }

    #[test]
    fn ecosystem_mapping() {
        assert_eq!(osv_ecosystem("cargo"), Some("crates.io"));
        assert_eq!(osv_ecosystem("pip"), Some("PyPI"));
        assert_eq!(osv_ecosystem("nuget"), None);
    }
}
