//! A Rig [`VectorStoreIndex`] over `core-store::hybrid_search` (sqlite-vec + FTS5, RRF-fused).
//! The query is embedded with the live `core-embed` embedder so vectors match the index.

use core_embed::Embedder;
use core_store::{KindFilter, RepoScope, Store};
use rig::vector_store::request::{Filter, VectorSearchRequest};
use rig::vector_store::{VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use std::sync::Arc;

/// A board-scoped retrieval index. `code = false` searches activity (commits/issues), `true` code.
/// `repo_ids` is the board's effective repo set; results never come from other repos.
pub struct RetrievalIndex {
    pub store: Arc<Store>,
    pub embedder: Arc<dyn Embedder>,
    pub repo_ids: Vec<String>,
    pub code: bool,
}

impl RetrievalIndex {
    /// Embed the query off the async runtime, then hybrid-search the right index slice.
    async fn search(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<core_store::EmbeddingHit>, VectorStoreError> {
        let embedder = self.embedder.clone();
        let q = query.to_string();
        let vector = tokio::task::spawn_blocking(move || embedder.embed(&q))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        let kind = if self.code {
            KindFilter::Is("code")
        } else {
            KindFilter::Not("code")
        };
        self.store
            .hybrid_search(&vector, query, n, kind, RepoScope::Repos(&self.repo_ids))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }
}

impl VectorStoreIndex for RetrievalIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let hits = self.search(req.query(), req.samples() as usize).await?;
        hits.into_iter()
            .map(|h| {
                let id = format!("{}:{}", h.ref_kind, h.ref_id);
                let doc = serde_json::json!({
                    "ref_kind": h.ref_kind,
                    "ref_id": h.ref_id,
                    "text": h.chunk,
                });
                let val: T = serde_json::from_value(doc).map_err(VectorStoreError::JsonError)?;
                Ok((h.score as f64, id, val))
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let hits = self.search(req.query(), req.samples() as usize).await?;
        Ok(hits
            .into_iter()
            .map(|h| (h.score as f64, format!("{}:{}", h.ref_kind, h.ref_id)))
            .collect())
    }
}
