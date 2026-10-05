//! Host-agnostic seams for talking to GitHub.
//! Auth and HTTP differ only at the edges, so the core depends on traits, not on a keychain or an
//! HTTP client. Conditional requests (ETag) are modeled so unchanged polls cost nothing against
//! the rate budget. The live `HttpTransport` (reqwest) is behind the off-by-default `http` feature.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Mutex;
use std::time::Duration;

/// Supplies a GitHub token. The desktop host backs this with the OS keychain; tests use
/// [`StaticTokenProvider`]. Returns `impl Future` rather than `async fn` to keep the bound `Send`.
pub trait TokenProvider {
    fn token(&self) -> impl Future<Output = Result<String, TokenError>> + Send;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TokenError {
    #[error("no token available")]
    Missing,
}

/// A token handed over directly (a pasted PAT, or a test fixture).
pub struct StaticTokenProvider(pub String);

impl TokenProvider for StaticTokenProvider {
    fn token(&self) -> impl Future<Output = Result<String, TokenError>> + Send {
        let token = self.0.clone();
        async move {
            if token.is_empty() {
                Err(TokenError::Missing)
            } else {
                Ok(token)
            }
        }
    }
}

/// Reads the token from an environment variable (a PAT in dev/CI).
pub struct EnvTokenProvider {
    var: String,
}

impl EnvTokenProvider {
    pub fn new(var: impl Into<String>) -> Self {
        Self { var: var.into() }
    }

    /// The conventional `GITHUB_TOKEN` variable.
    pub fn github() -> Self {
        Self::new("GITHUB_TOKEN")
    }
}

impl TokenProvider for EnvTokenProvider {
    fn token(&self) -> impl Future<Output = Result<String, TokenError>> + Send {
        let value = std::env::var(&self.var).ok().filter(|s| !s.is_empty());
        async move { value.ok_or(TokenError::Missing) }
    }
}

/// HTTP verb. Only GET is used for polling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
}

/// A request to GitHub. `etag` becomes the `If-None-Match` header when set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GhRequest {
    pub method: Method,
    pub path: String,
    pub etag: Option<String>,
}

/// A response from GitHub, reduced to what the conditional-GET client needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GhResponse {
    pub status: u16,
    pub etag: Option<String>,
    pub body: String,
    /// `x-ratelimit-remaining`, when present.
    pub rate_limit_remaining: Option<u32>,
}

/// Carries GitHub requests to the network. The desktop host backs this with reqwest; tests use
/// [`FakeTransport`].
pub trait Transport {
    fn send(
        &self,
        request: GhRequest,
    ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send;

    /// POST a JSON body to `path` (the GraphQL endpoint, for data REST does not expose, such as a
    /// PR's closing-issue references). Default: unsupported; only the live reqwest transport and
    /// GraphQL tests override it.
    fn post_json(
        &self,
        path: String,
        body: String,
    ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send {
        let _ = (path, body);
        async {
            Err(TransportError::Failed(
                "post_json not supported by this transport".into(),
            ))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TransportError {
    #[error("transport failed: {0}")]
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GithubError {
    #[error(transparent)]
    Token(#[from] TokenError),
    #[error(transparent)]
    Transport(#[from] TransportError),
    /// The forge rejected our credentials (HTTP 401): token missing, expired or revoked.
    #[error("authentication failed (401)")]
    Unauthorized,
    /// The rate limit is exhausted (HTTP 403/429 with `x-ratelimit-remaining: 0`).
    #[error("rate limited")]
    RateLimited,
    #[error("unexpected status {0}")]
    Status(u16),
    /// A 304 arrived but no cached body was held for that path.
    #[error("got 304 for `{0}` with no cached body")]
    StaleNotModified(String),
}

/// The outcome of a conditional GET.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchOutcome {
    pub body: String,
    /// True when the server answered 304 and the cached body was served.
    pub unchanged: bool,
}

/// The outcome of [`GithubClient::get_with_etag`]: a conditional GET on a caller-persisted ETag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalFetch {
    /// The fresh body on 200; `None` on 304 (the caller's stored rows are still current).
    pub body: Option<String>,
    /// The ETag to persist for next time. On 304 this is the ETag the caller supplied.
    pub etag: Option<String>,
}

struct CacheEntry {
    etag: String,
    body: String,
}

/// Issues conditional GETs against a [`Transport`], caching ETags so unchanged polls are cheap.
pub struct GithubClient<T, P> {
    transport: T,
    tokens: P,
    cache: Mutex<HashMap<String, CacheEntry>>,
    rate_limit_remaining: Mutex<Option<u32>>,
}

impl<T: Transport, P: TokenProvider> GithubClient<T, P> {
    pub fn new(transport: T, tokens: P) -> Self {
        Self {
            transport,
            tokens,
            cache: Mutex::new(HashMap::new()),
            rate_limit_remaining: Mutex::new(None),
        }
    }

    /// The latest rate-limit budget seen on a response.
    pub fn rate_limit_remaining(&self) -> Option<u32> {
        *self.rate_limit_remaining.lock().unwrap()
    }

    /// POST a GraphQL request body to `/graphql` and return the raw response body. The transport
    /// attaches the token, as on the GET path.
    pub async fn graphql(&self, request_body: String) -> Result<String, GithubError> {
        let _token = self.tokens.token().await?;
        let response = self
            .transport
            .post_json("/graphql".to_string(), request_body)
            .await?;
        if response.status == 200 {
            Ok(response.body)
        } else {
            Err(GithubError::Status(response.status))
        }
    }

    /// GET `path` conditionally: send the cached ETag as `If-None-Match`, serve the cache on 304,
    /// refresh on 200.
    pub async fn get_conditional(&self, path: &str) -> Result<FetchOutcome, GithubError> {
        // Fetched per call so the host can rotate the token.
        let _token = self.tokens.token().await?;

        let etag = self.cache.lock().unwrap().get(path).map(|e| e.etag.clone());

        let response = self
            .transport
            .send(GhRequest {
                method: Method::Get,
                path: path.to_string(),
                etag,
            })
            .await?;

        if let Some(remaining) = response.rate_limit_remaining {
            *self.rate_limit_remaining.lock().unwrap() = Some(remaining);
        }

        match response.status {
            304 => {
                let cache = self.cache.lock().unwrap();
                let entry = cache
                    .get(path)
                    .ok_or_else(|| GithubError::StaleNotModified(path.to_string()))?;
                Ok(FetchOutcome {
                    body: entry.body.clone(),
                    unchanged: true,
                })
            }
            200 => {
                if let Some(etag) = response.etag {
                    self.cache.lock().unwrap().insert(
                        path.to_string(),
                        CacheEntry {
                            etag,
                            body: response.body.clone(),
                        },
                    );
                }
                Ok(FetchOutcome {
                    body: response.body,
                    unchanged: false,
                })
            }
            other => Err(classify_error(other, response.rate_limit_remaining)),
        }
    }

    /// GET `path` conditionally against a caller-supplied ETag (e.g. persisted across restarts).
    /// Returns the fresh body and new ETag on 200, or `body: None` on 304. Unlike
    /// [`get_conditional`](Self::get_conditional) it keeps no in-memory body cache. Used for
    /// low-volume entities (releases, CI runs, dep manifests) whose ETag lives in `sync_cursors`.
    pub async fn get_with_etag(
        &self,
        path: &str,
        etag: Option<String>,
    ) -> Result<ConditionalFetch, GithubError> {
        // Fetched per call so the host can rotate the token.
        let _token = self.tokens.token().await?;

        let response = self
            .transport
            .send(GhRequest {
                method: Method::Get,
                path: path.to_string(),
                etag: etag.clone(),
            })
            .await?;

        if let Some(remaining) = response.rate_limit_remaining {
            *self.rate_limit_remaining.lock().unwrap() = Some(remaining);
        }

        match response.status {
            // Unchanged: no body, keep the ETag the caller already holds.
            304 => Ok(ConditionalFetch { body: None, etag }),
            // Changed: return the body and advance to the new ETag (keep the old if none sent).
            200 => Ok(ConditionalFetch {
                body: Some(response.body),
                etag: response.etag.or(etag),
            }),
            other => Err(classify_error(other, response.rate_limit_remaining)),
        }
    }
}

/// Classify a non-OK/304 status into a typed error: 401 is an auth failure; 403/429 with the
/// rate-limit budget exhausted is a rate limit; anything else is a generic status.
fn classify_error(status: u16, rate_limit_remaining: Option<u32>) -> GithubError {
    match status {
        401 => GithubError::Unauthorized,
        403 | 429 if rate_limit_remaining == Some(0) => GithubError::RateLimited,
        other => GithubError::Status(other),
    }
}

/// Stagger `n` sources across `interval` so they do not poll at the same instant. Offset of
/// source `i` is `interval * i / n`.
pub fn staggered_offsets(n: usize, interval: Duration) -> Vec<Duration> {
    if n == 0 {
        return Vec::new();
    }
    (0..n).map(|i| interval * i as u32 / n as u32).collect()
}

/// Live GitHub transport over reqwest (`http` feature). Conditional caching stays in
/// `GithubClient`.
#[cfg(feature = "http")]
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    token: String,
}

#[cfg(feature = "http")]
impl HttpTransport {
    /// Build a transport for `base_url` (e.g. "https://api.github.com") using `token`.
    pub fn new(
        base_url: impl Into<String>,
        token: impl Into<String>,
    ) -> Result<Self, TransportError> {
        let client = reqwest::Client::builder()
            .user_agent("orgonzola")
            .build()
            .map_err(|e| TransportError::Failed(e.to_string()))?;
        Ok(Self {
            client,
            base_url: base_url.into(),
            token: token.into(),
        })
    }
}

#[cfg(feature = "http")]
impl Transport for HttpTransport {
    fn send(
        &self,
        request: GhRequest,
    ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send {
        let url = format!("{}{}", self.base_url, request.path);
        let token = self.token.clone();
        let client = self.client.clone();
        async move {
            let mut req = client
                .get(&url)
                .header("Authorization", format!("Bearer {token}"))
                .header("Accept", "application/vnd.github+json");
            if let Some(etag) = &request.etag {
                req = req.header("If-None-Match", etag.clone());
            }
            let resp = req
                .send()
                .await
                .map_err(|e| TransportError::Failed(e.to_string()))?;
            let status = resp.status().as_u16();
            let header = |name: &str| {
                resp.headers()
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string())
            };
            let etag = header("etag");
            let rate_limit_remaining = header("x-ratelimit-remaining").and_then(|s| s.parse().ok());
            let body = resp
                .text()
                .await
                .map_err(|e| TransportError::Failed(e.to_string()))?;
            Ok(GhResponse {
                status,
                etag,
                body,
                rate_limit_remaining,
            })
        }
    }

    fn post_json(
        &self,
        path: String,
        body: String,
    ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send {
        let url = format!("{}{}", self.base_url, path);
        let token = self.token.clone();
        let client = self.client.clone();
        async move {
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .body(body)
                .send()
                .await
                .map_err(|e| TransportError::Failed(e.to_string()))?;
            let status = resp.status().as_u16();
            let body = resp
                .text()
                .await
                .map_err(|e| TransportError::Failed(e.to_string()))?;
            Ok(GhResponse {
                status,
                etag: None,
                body,
                rate_limit_remaining: None,
            })
        }
    }
}

// ---- GitHub OAuth device flow ----
// The device flow needs no client secret and no redirect listener. These are host-agnostic HTTP
// primitives; the endpoint base is configurable so a wiremock server can stand in for github.com.

/// The OAuth host for the device flow (github.com, not api.github.com). Tests pass a mock URL.
pub const GITHUB_OAUTH_BASE: &str = "https://github.com";

/// A started device authorization. Show `user_code`, send the user to `verification_uri`, then
/// poll with `device_code` every `interval_secs` until it resolves or `expires_in_secs` elapses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub interval_secs: u64,
    pub expires_in_secs: u64,
}

/// The outcome of one device-token poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevicePoll {
    /// Still waiting for the user to authorize; poll again after the interval.
    Pending,
    /// Polling too fast; lengthen the interval and try again.
    SlowDown,
    /// Authorized: the access token to store (in the keychain, never the DB).
    Authorized(String),
    /// The user denied the request.
    Denied,
    /// The device code expired before authorization; restart the flow.
    Expired,
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceFlowError {
    #[error("device-flow request failed: {0}")]
    Transport(String),
    #[error("device-flow response was not understood: {0}")]
    Decode(String),
}

/// Start the device flow: request a device and user code for `client_id` with the OAuth `scope`
/// (e.g. "repo read:org"). `base_url` is [`GITHUB_OAUTH_BASE`] in production.
#[cfg(feature = "http")]
pub async fn request_device_code(
    base_url: &str,
    client_id: &str,
    scope: &str,
) -> Result<DeviceCode, DeviceFlowError> {
    #[derive(serde::Deserialize)]
    struct Resp {
        device_code: String,
        user_code: String,
        verification_uri: String,
        #[serde(default)]
        interval: u64,
        #[serde(default)]
        expires_in: u64,
    }
    let url = format!("{}/login/device/code", base_url.trim_end_matches('/'));
    let body = oauth_post(&url, &[("client_id", client_id), ("scope", scope)]).await?;
    let r: Resp = serde_json::from_str(&body)
        .map_err(|e| DeviceFlowError::Decode(format!("device code: {e}: {body}")))?;
    Ok(DeviceCode {
        device_code: r.device_code,
        user_code: r.user_code,
        verification_uri: r.verification_uri,
        interval_secs: r.interval.max(1),
        expires_in_secs: if r.expires_in == 0 { 900 } else { r.expires_in },
    })
}

/// Poll once for the access token. The caller loops on `Pending`/`SlowDown`.
#[cfg(feature = "http")]
pub async fn poll_device_token(
    base_url: &str,
    client_id: &str,
    device_code: &str,
) -> Result<DevicePoll, DeviceFlowError> {
    #[derive(serde::Deserialize)]
    struct Resp {
        access_token: Option<String>,
        error: Option<String>,
    }
    let url = format!(
        "{}/login/oauth/access_token",
        base_url.trim_end_matches('/')
    );
    let body = oauth_post(
        &url,
        &[
            ("client_id", client_id),
            ("device_code", device_code),
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
        ],
    )
    .await?;
    // Do not embed the response body in the error: a malformed-but-token-bearing 200 could carry
    // the access token into a UI-visible error string.
    let r: Resp = serde_json::from_str(&body)
        .map_err(|e| DeviceFlowError::Decode(format!("device token response: {e}")))?;
    if let Some(token) = r.access_token.filter(|t| !t.is_empty()) {
        return Ok(DevicePoll::Authorized(token));
    }
    Ok(match r.error.as_deref() {
        Some("authorization_pending") => DevicePoll::Pending,
        Some("slow_down") => DevicePoll::SlowDown,
        Some("access_denied") => DevicePoll::Denied,
        Some("expired_token") => DevicePoll::Expired,
        other => {
            return Err(DeviceFlowError::Decode(format!(
                "unexpected device-token response (error={})",
                other.unwrap_or("<none>")
            )))
        }
    })
}

/// POST a form to an OAuth endpoint asking for a JSON response, returning the raw body text.
#[cfg(feature = "http")]
async fn oauth_post(url: &str, form: &[(&str, &str)]) -> Result<String, DeviceFlowError> {
    let client = reqwest::Client::builder()
        .user_agent("orgonzola")
        .build()
        .map_err(|e| DeviceFlowError::Transport(e.to_string()))?;
    client
        .post(url)
        .header("Accept", "application/json")
        .form(form)
        .send()
        .await
        .map_err(|e| DeviceFlowError::Transport(e.to_string()))?
        .text()
        .await
        .map_err(|e| DeviceFlowError::Transport(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn classify_error_distinguishes_auth_and_rate_limit() {
        assert_eq!(classify_error(401, Some(5000)), GithubError::Unauthorized);
        assert_eq!(classify_error(403, Some(0)), GithubError::RateLimited);
        assert_eq!(classify_error(429, Some(0)), GithubError::RateLimited);
        // 403 with budget left is a permissions issue, not rate-limit -> generic status.
        assert_eq!(classify_error(403, Some(120)), GithubError::Status(403));
        assert_eq!(classify_error(500, None), GithubError::Status(500));
    }

    /// Records requests it receives and returns programmed responses in order.
    struct FakeTransport {
        responses: Mutex<VecDeque<GhResponse>>,
        seen: Mutex<Vec<GhRequest>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<GhResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    impl Transport for FakeTransport {
        fn send(
            &self,
            request: GhRequest,
        ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send {
            self.seen.lock().unwrap().push(request);
            let next = self.responses.lock().unwrap().pop_front();
            async move { next.ok_or_else(|| TransportError::Failed("no programmed response".into())) }
        }
    }

    fn resp(status: u16, etag: Option<&str>, body: &str, remaining: u32) -> GhResponse {
        GhResponse {
            status,
            etag: etag.map(|s| s.to_string()),
            body: body.to_string(),
            rate_limit_remaining: Some(remaining),
        }
    }

    #[tokio::test]
    async fn conditional_get_caches_then_serves_304_with_etag() {
        let transport = FakeTransport::new(vec![
            resp(200, Some("v1"), "ALPHA", 4999),
            resp(304, Some("v1"), "", 4999),
        ]);
        let client = GithubClient::new(transport, StaticTokenProvider("pat".into()));

        let first = client
            .get_conditional("/repos/acme/widget/commits")
            .await
            .unwrap();
        assert_eq!(first.body, "ALPHA");
        assert!(!first.unchanged);

        let second = client
            .get_conditional("/repos/acme/widget/commits")
            .await
            .unwrap();
        assert_eq!(second.body, "ALPHA");
        assert!(second.unchanged);

        // The second request carries the cached ETag.
        let seen = client.transport.seen.lock().unwrap();
        assert_eq!(seen[0].etag, None);
        assert_eq!(seen[1].etag, Some("v1".to_string()));
        assert_eq!(client.rate_limit_remaining(), Some(4999));
    }

    #[tokio::test]
    async fn conditional_get_refreshes_cache_on_200() {
        let transport = FakeTransport::new(vec![
            resp(200, Some("v1"), "ALPHA", 4999),
            resp(200, Some("v2"), "BETA", 4998),
        ]);
        let client = GithubClient::new(transport, StaticTokenProvider("pat".into()));

        client.get_conditional("/x").await.unwrap();
        let second = client.get_conditional("/x").await.unwrap();
        assert_eq!(second.body, "BETA");
        assert!(!second.unchanged);

        let seen = client.transport.seen.lock().unwrap();
        // Second request carried the v1 ETag; the server changed it, so the cache now holds v2.
        assert_eq!(seen[1].etag, Some("v1".to_string()));
    }

    #[tokio::test]
    async fn get_with_etag_returns_body_and_new_etag_on_200() {
        let transport = FakeTransport::new(vec![resp(200, Some("v1"), "ALPHA", 4999)]);
        let client = GithubClient::new(transport, StaticTokenProvider("pat".into()));

        let out = client.get_with_etag("/x", None).await.unwrap();
        assert_eq!(out.body.as_deref(), Some("ALPHA"));
        assert_eq!(out.etag, Some("v1".to_string()));
        // No ETag yet, so no If-None-Match.
        assert_eq!(client.transport.seen.lock().unwrap()[0].etag, None);
    }

    #[tokio::test]
    async fn get_with_etag_forwards_supplied_etag_and_reports_304_unchanged() {
        // A fresh client with no in-memory cache must still send a caller-persisted ETag.
        let transport = FakeTransport::new(vec![resp(304, None, "", 4999)]);
        let client = GithubClient::new(transport, StaticTokenProvider("pat".into()));

        let out = client
            .get_with_etag("/x", Some("v1".to_string()))
            .await
            .unwrap();
        // 304: no body, the supplied ETag is handed back to persist again.
        assert_eq!(out.body, None);
        assert_eq!(out.etag, Some("v1".to_string()));
        assert_eq!(
            client.transport.seen.lock().unwrap()[0].etag,
            Some("v1".to_string())
        );
    }

    #[tokio::test]
    async fn empty_token_is_rejected() {
        let transport = FakeTransport::new(vec![]);
        let client = GithubClient::new(transport, StaticTokenProvider(String::new()));
        let err = client.get_conditional("/x").await.unwrap_err();
        assert_eq!(err, GithubError::Token(TokenError::Missing));
    }

    #[tokio::test]
    async fn env_token_provider_reads_or_reports_missing() {
        // A unique var name so this does not race other tests sharing the process environment.
        let var = "ORGONZOLA_ENVTOK_TEST_VAR";
        let provider = EnvTokenProvider::new(var);
        std::env::remove_var(var);
        assert_eq!(provider.token().await, Err(TokenError::Missing));
        std::env::set_var(var, "ghp_xyz");
        assert_eq!(provider.token().await, Ok("ghp_xyz".to_string()));
        std::env::remove_var(var);
    }

    #[test]
    fn offsets_are_evenly_staggered() {
        let offsets = staggered_offsets(4, Duration::from_secs(60));
        assert_eq!(
            offsets,
            vec![
                Duration::from_secs(0),
                Duration::from_secs(15),
                Duration::from_secs(30),
                Duration::from_secs(45),
            ]
        );
        assert!(staggered_offsets(0, Duration::from_secs(60)).is_empty());
    }
}

#[cfg(all(test, feature = "http"))]
mod http_tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn http_transport_sends_auth_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/repos/acme/widget/commits"))
            .and(header("authorization", "Bearer pat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("etag", "\"v1\"")
                    .insert_header("x-ratelimit-remaining", "4321")
                    .set_body_string("[]"),
            )
            .mount(&server)
            .await;

        let transport = HttpTransport::new(server.uri(), "pat").unwrap();
        let resp = transport
            .send(GhRequest {
                method: Method::Get,
                path: "/repos/acme/widget/commits".into(),
                etag: None,
            })
            .await
            .unwrap();

        assert_eq!(resp.status, 200);
        assert_eq!(resp.etag.as_deref(), Some("\"v1\""));
        assert_eq!(resp.rate_limit_remaining, Some(4321));
        assert_eq!(resp.body, "[]");
    }

    #[tokio::test]
    async fn device_flow_requests_a_code() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .and(header("accept", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"device_code":"dev123","user_code":"ABCD-1234","verification_uri":"https://github.com/login/device","expires_in":900,"interval":5}"#,
            ))
            .mount(&server)
            .await;

        let code = request_device_code(&server.uri(), "client-xyz", "repo")
            .await
            .unwrap();
        assert_eq!(code.device_code, "dev123");
        assert_eq!(code.user_code, "ABCD-1234");
        assert_eq!(code.verification_uri, "https://github.com/login/device");
        assert_eq!(code.interval_secs, 5);
        assert_eq!(code.expires_in_secs, 900);
    }

    /// Poll a token endpoint that returns `body`, asserting it maps to `expected`.
    async fn poll_with(body: &str) -> DevicePoll {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;
        poll_device_token(&server.uri(), "client-xyz", "dev123")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn device_flow_poll_maps_each_outcome() {
        assert_eq!(
            poll_with(r#"{"access_token":"gho_abc","token_type":"bearer"}"#).await,
            DevicePoll::Authorized("gho_abc".into())
        );
        assert_eq!(
            poll_with(r#"{"error":"authorization_pending"}"#).await,
            DevicePoll::Pending
        );
        assert_eq!(
            poll_with(r#"{"error":"slow_down","interval":10}"#).await,
            DevicePoll::SlowDown
        );
        assert_eq!(
            poll_with(r#"{"error":"access_denied"}"#).await,
            DevicePoll::Denied
        );
        assert_eq!(
            poll_with(r#"{"error":"expired_token"}"#).await,
            DevicePoll::Expired
        );
    }
}
