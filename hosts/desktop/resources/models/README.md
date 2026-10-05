# Bundled ML models

The embedding and reranker models the shell bundles as an app resource. The files are not committed
(about 200 MB). Fetch them before building or running with the `fastembed` feature:

```
just fetch-model
```

This creates `jina-embeddings-v2-base-code/` and `jina-reranker-v1-turbo-en/`, each holding an ONNX
graph and tokenizer files. The bundle maps this directory to `models/` inside the app, and the shell
loads from there at runtime; it never downloads models. Without the `fastembed` feature the shell uses
the deterministic `HashEmbedder` and ignores this directory.
