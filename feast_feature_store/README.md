# Feast Feature Store

Feast is used to define, store, and serve the **29 features** used by the Learning-to-Rank (LTR) model. Offline feature data is written to S3 by the recommendation service; the online store (Redis) is used at request time for LTR scoring.

## Functionality

- **Entity**: `user_book` — unique (user_id, book_id) pair.
- **Feature view**: `user_book_features` — 29 features (retrieval scores, genre/author/language match, ratings, social, session, device, popularity, diversity). Defined in `feature_repo/feature_definitions.py` with a Parquet source on S3.
- **Offline store**: Parquet files under `s3://bibliophile-ai-feast/features/.../user_book_features.parquet`, produced by the recommendation service’s feature engineering step when building LTR candidates.
- **Online store**: Redis (e.g. `feast-redis:6379` in K8s). The recommendation service materializes from offline to online so that `get_online_features` returns low-latency feature vectors for (user_book) keys at inference time.
- **Retrieve**: `retrieve_features.py` shows how to call `store.get_online_features()` for a list of `user_book` entity keys; the recommendation service uses this (or in-memory features when the online store is empty) inside `ltr_ranking.py`.

## Implementation in this project

- **feature_repo/feature_definitions.py**: Feast Entity `user_book`, FileSource (Parquet on S3), FeatureView `user_book_features` with 29 Field definitions.
- **feature_repo/feature_store.yaml**: Project name, S3 registry path, Redis online store connection, file offline store.
- **retrieve_features.py**: Script/helper to load FeatureStore and call `get_online_features` for given entity rows; used as reference by the recommendation service.
- **Recommendation service**: `feature_engineering/feature_service.py` builds the 29 features from `combined_scores_map` and session context, writes to Feast offline store (S3), and runs materialization to Redis. LTR training (`ltr_train`) reads from S3/Feast for labels+features; LTR inference uses online features or in-memory fallback.

## Usage

- **Training**: LTR training job uses Feast repo (or direct S3 parquet) to join labels with the 29 features.
- **Inference**: Recommendation service calls Feast online store for (user_id, book_id) pairs; if values are missing (e.g. before materialization), it falls back to in-memory features computed in the same request.
