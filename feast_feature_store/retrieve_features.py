from feast import FeatureStore
import pandas as pd
from datetime import datetime

# Initialize Feature Store
store = FeatureStore(repo_path="./feature_repo")

def get_online_features(entity_keys):
    """
    Retrieve features from Redis for real-time inference
    
    Args:
        entity_keys: list of user_book entity IDs
    """
    # Create entity dataframe
    entity_df = pd.DataFrame({
        "user_book": entity_keys
    })
    
    # Get features from Redis (online store)
    feature_vector = store.get_online_features(
        entity_rows=entity_df.to_dict('records'),
        features=[
            "user_book_features:query_id",
            "user_book_features:retrieval_0",
            "user_book_features:retrieval_1",
            "user_book_features:retrieval_2",
            "user_book_features:retrieval_3",
            "user_book_features:retrieval_4",
            "user_book_features:retrieval_5",
            "user_book_features:genre_match_count",
            "user_book_features:author_match",
            "user_book_features:language_match",
            "user_book_features:avg_rating",
            "user_book_features:user_rating",
            "user_book_features:avg_rating_diff",
            "user_book_features:rating_count",
            "user_book_features:user_pref_strength",
            "user_book_features:friend_reads_count",
            "user_book_features:friend_avg_rating",
            "user_book_features:author_following",
            "user_book_features:mutual_likes",
            "user_book_features:social_proximity",
            "user_book_features:session_position",
            "user_book_features:session_genre_drift",
            "user_book_features:time_since_last_action",
            "user_book_features:is_mobile",
            "user_book_features:is_desktop",
            "user_book_features:is_tablet",
            "user_book_features:session_length",
            "user_book_features:global_pop_rank",
            "user_book_features:trending_score",
            "user_book_features:intra_list_diversity",
        ]
    ).to_df()
    
    return feature_vector

def get_historical_features(entity_df):
    """
    Retrieve features from S3 for training
    
    Args:
        entity_df: DataFrame with 'user_book' and 'timestamp' columns
    """
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_book_features:query_id",
            "user_book_features:retrieval_0",
            "user_book_features:retrieval_1",
            "user_book_features:retrieval_2",
            "user_book_features:retrieval_3",
            "user_book_features:retrieval_4",
            "user_book_features:retrieval_5",
            "user_book_features:genre_match_count",
            "user_book_features:author_match",
            "user_book_features:language_match",
            "user_book_features:avg_rating",
            "user_book_features:user_rating",
            "user_book_features:avg_rating_diff",
            "user_book_features:rating_count",
            "user_book_features:user_pref_strength",
            "user_book_features:friend_reads_count",
            "user_book_features:friend_avg_rating",
            "user_book_features:author_following",
            "user_book_features:mutual_likes",
            "user_book_features:social_proximity",
            "user_book_features:session_position",
            "user_book_features:session_genre_drift",
            "user_book_features:time_since_last_action",
            "user_book_features:is_mobile",
            "user_book_features:is_desktop",
            "user_book_features:is_tablet",
            "user_book_features:session_length",
            "user_book_features:global_pop_rank",
            "user_book_features:trending_score",
            "user_book_features:intra_list_diversity",
        ]
    ).to_df()
    
    return training_df

if __name__ == "__main__":
    # Real-time inference
    print("Fetching online features from Redis...")
    online_features = get_online_features([
        "00cf6f15-d2e3-4ec3-ae97-081a45f79111_58997",
        "00cf6f15-d2e3-4ec3-ae97-081a45f79111_37635",
        "00cf6f15-d2e3-4ec3-ae97-081a45f79111_245"
    ])
    print(online_features)
    print(f"\nRetrieved {len(online_features)} feature vectors")
    
    # Historical features for training
    print("\nFetching historical features from S3...")
    entity_df = pd.DataFrame({
        'user_book': [
            '00cf6f15-d2e3-4ec3-ae97-081a45f79111_67846',
            '00cf6f15-d2e3-4ec3-ae97-081a45f79111_1454'
        ],
        'event_timestamp': [
            datetime(2026, 1, 13, 18, 12, 38),
            datetime(2026, 1, 13, 18, 12, 38)
        ]
    })
    
    historical_features = get_historical_features(
        entity_df
    )
    print(historical_features)