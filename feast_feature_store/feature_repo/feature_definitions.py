from feast import FeatureView, Entity, Feature, FileSource, PushSource, ValueType, Field
from feast.data_format import ParquetFormat
from feast.types import Float32, Int32, String
from datetime import timedelta

# Entity (unique user-book pair)
user_book = Entity(
    name="user_book",
    value_type=ValueType.STRING,
)

# Offline Parquet source (S3)
book_features_parquet = FileSource(
    file_format=ParquetFormat(),
    path="s3://bibliophile-ai-feast/features/*/*/*/user_book_features.parquet",  
    timestamp_field="timestamp"
)

# MAIN 29-FEATURE VIEW
user_book_features = FeatureView(
    name="user_book_features",
    entities=[user_book],
    ttl=timedelta(days=7),
    schema=[
        Field(name="query_id", dtype=String),
        Field(name="user_id", dtype=String),
        Field(name="book_id", dtype=String),  
        Field(name="retrieval_0", dtype=Float32),
        Field(name="retrieval_1", dtype=Float32),
        Field(name="retrieval_2", dtype=Float32),
        Field(name="retrieval_3", dtype=Float32),
        Field(name="retrieval_4", dtype=Float32),
        Field(name="retrieval_5", dtype=Float32),
        Field(name="genre_match_count", dtype=Int32),
        Field(name="author_match", dtype=Int32),
        Field(name="language_match", dtype=Int32),
        Field(name="avg_rating", dtype=Float32),
        Field(name="user_rating", dtype=Float32),
        Field(name="avg_rating_diff", dtype=Float32),
        Field(name="rating_count", dtype=Int32),
        Field(name="user_pref_strength", dtype=Float32),
        Field(name="friend_reads_count", dtype=Int32),
        Field(name="friend_avg_rating", dtype=Float32),
        Field(name="author_following", dtype=Int32),
        Field(name="mutual_likes", dtype=Int32),
        Field(name="social_proximity", dtype=Int32),
        Field(name="session_position", dtype=Int32),
        Field(name="session_genre_drift", dtype=Float32),
        Field(name="time_since_last_action", dtype=Float32),
        Field(name="is_mobile", dtype=Int32),
        Field(name="is_desktop", dtype=Int32),
        Field(name="is_tablet", dtype=Int32),
        Field(name="session_length", dtype=Int32),
        Field(name="global_pop_rank", dtype=Float32),
        Field(name="trending_score", dtype=Float32),
        Field(name="intra_list_diversity", dtype=Float32),
    ],
    source=book_features_parquet,
    tags={"team": "recsys"},
    online=True
)
