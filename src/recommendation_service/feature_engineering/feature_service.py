# feature_service.py - COMPLETE 29-feature pipeline with ALL corrections
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
import supabase
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from neo4j import GraphDatabase
import redis
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

class FeatureService:
    def __init__(self):
        """Initialize all data source connections"""
        # Supabase PostgreSQL
        self.supabase = supabase.create_client(
            os.environ["SUPABASE_URL"], 
            os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        )
        
        # MongoDB Clickstream
        self.mongo_client = MongoClient(
            os.environ["MONGO_URI"], 
            server_api=ServerApi('1'), 
            serverSelectionTimeoutMS=5000
        )
        self.events_col = self.mongo_client["click_stream"]["events"]
        
        # Neo4j Graph
        self.neo4j_driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        )
        
        # Redis
        self.redis_client = redis.from_url(os.environ["REDIS_URL"])
        
        # Cache
        self.user_cache = {}
        self.books_cache = {}
        self.s3_bucket = "bibliophile-ai-feast"
        self.s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,)
        self.parquet_path = f"features/{datetime.now().strftime('%Y/%m/%d')}/user_book_features.parquet"

    async def store_features_s3_parquet(self, feature_entities: List[Dict]) -> None:
        """Append features to S3 Parquet - FIXED buffer conversion"""
        if not feature_entities:
            return
            
        try:
            df_new = pd.DataFrame(feature_entities)
            
            # Check if exists
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=self.parquet_path)
                exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    exists = False
                else:
                    raise e
            
            # Download + append
            if exists:
                obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.parquet_path)
                existing_df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            else:
                existing_df = pd.DataFrame()
            
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
            
            buffer = io.BytesIO()
            df_combined.to_parquet(buffer, index=False)
            parquet_bytes = buffer.getvalue() 
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=self.parquet_path,
                Body=parquet_bytes, 
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Appended {len(df_new)} → Total: {len(df_combined)}")
            print(f"Appended {len(df_new)} → Total: {len(df_combined)}")
            
        except Exception as e:
            logger.error(f"S3 Parquet failed: {e}")
            print(f"S3 Parquet failed: {e}")

    
    async def engineer_features_batch(
        self,
        user_id: str,
        book_ids: List[str],
        combined_scores_map: Dict[str, Dict[str, float]],  
        session_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate 29-feature vectors from combined_scores_map"""
        logger.info(f"Engineering {len(book_ids)} books for user {user_id}")
        
        # Batch fetch ALL data
        books_meta = await self._batch_get_books_metadata(book_ids)
        user_profile = await self._get_user_profile(user_id)
        ratings = await self._get_ratings(user_id, book_ids)
        session_info = await self._get_current_session(user_id)
        recent_reads = await self._get_recent_reads(user_id)
        social_features = await self._batch_social_features(user_id, book_ids)
        pop_features = await self._batch_popularity_features(book_ids)
        
        # Build features
        feature_entities = []
        query_id = str(uuid.uuid4())
        for i, book_id in enumerate(book_ids):
            try:
                fv = self._build_single_feature_vector(
                    user_id, book_id, combined_scores_map, session_context,
                    books_meta.get(book_id, {}), user_profile, ratings, session_info,
                    recent_reads, social_features.get(book_id, {}), 
                    pop_features.get(book_id, {}), position=i, query_id=query_id
                )
                feature_entities.append(fv)
            except Exception as e:
                logger.error(f"Feature error {book_id}: {e}")
        
        logger.info(f"Generated {len(feature_entities)} / {len(book_ids)} features")
        await self.store_features_s3_parquet(feature_entities)
        return feature_entities
    
    def _build_single_feature_vector(
        self, user_id: str, book_id: str, combined_scores_map: Dict[str, Dict],
        session_context: Dict, book_meta: Dict, user_profile: Dict, ratings: Dict,
        session_info: Dict, recent_reads: List[str], social_feats: Dict, 
        pop_feats: Dict, position: int, query_id: str
    ) -> Dict[str, Any]:
        """Handle list authors/categories + combined_scores_map"""
        
        # 1. RETRIEVAL SCORES (6) - FIXED combined_scores_map format
        book_scores = combined_scores_map.get(book_id, {})
        retrieval_scores = [
            book_scores.get("content_score", 0.0),    # cbf
            book_scores.get("als_score", 0.0),        # cf  
            book_scores.get("graph_score", 0.0),      # graph
            book_scores.get("session_score", 0.0),    # seq
            book_scores.get("popularity_score", 0.0), # pop
            book_scores.get("linucb_score", 0.5)      # linucb
        ]
        
        # 2. USER-BOOK METADATA (8) - FIXED LIST HANDLING
        book_authors = book_meta.get("authors", [])  # LIST
        book_categories = book_meta.get("categories", [])  # LIST
        user_genres = user_profile.get("genres", [])  # LIST
        user_authors = user_profile.get("authors", [])  # LIST
        
        # Genre overlap (set intersection for lists)
        genre_match_count = len(set(book_categories) & set(user_genres))
        
        # Author match (any overlap)
        author_match = 1 if any(author in user_authors for author in book_authors) else 0
        
        language_match = 1 if book_meta.get("language") == user_profile.get("preferred_language", "en") else 0
        book_ratings = ratings.get(book_id, {"user_rating": 0.0, "avg_rating": 3.5, "rating_count": 0})
        # User rating from book metadata
        user_rating = book_ratings["user_rating"]   
        avg_rating = book_ratings["avg_rating"] 
        rating_count = book_ratings["rating_count"] 
        
        user_pref_strength = min(1.0, genre_match_count / max(1, len(book_categories)))
        
        # 3. SOCIAL FEATURES (5)
        friend_reads_count = social_feats.get("friend_reads", 0)
        friend_avg_rating = social_feats.get("friend_avg_rating", 0.0)
        author_following = social_feats.get("author_following", 0)
        mutual_likes = social_feats.get("mutual_likes", 0)
        social_proximity = social_feats.get("social_proximity", 999)
        
        # 4. SESSION CONTEXT (7) - FIXED from MongoDB
        session_position = position + 1
        session_genre_drift = self._genre_drift(user_genres, book_categories)
        time_since_last_action = session_info.get("time_since_last_action", 0.0)
        
        device = session_context.get("device", "desktop")
        is_mobile = 1 if device == "mobile" else 0
        is_desktop = 1 if device == "desktop" else 0
        is_tablet = 1 if device == "tablet" else 0
        session_length = session_info.get("session_action_count", 1)
        
        # 5. POPULARITY (3)
        global_pop_rank = pop_feats.get("global_rank", 0.5)
        trending_score = pop_feats.get("trending_score", 0.0)
        intra_list_diversity = pop_feats.get("diversity", 0.5)
        
        # FULL 29-DIM VECTOR
        feature_vector = {
            "user_book": f"{user_id}_{book_id}",
            "user_id": user_id,
            "book_id": book_id,
            "query_id": query_id,
            "timestamp": datetime.utcnow(),
            
            # 29 Features 
            **{f"retrieval_{i}": float(retrieval_scores[i]) for i in range(6)},
            "genre_match_count": int(genre_match_count),
            "author_match": int(author_match),
            "language_match": int(language_match),
            "avg_rating": float(avg_rating),
            "user_rating": float(user_rating),
            "avg_rating_diff": float(user_rating - avg_rating),
            "rating_count": int(rating_count),
            "user_pref_strength": float(user_pref_strength),
            "friend_reads_count": int(friend_reads_count),
            "friend_avg_rating": float(friend_avg_rating),
            "author_following": int(author_following),
            "mutual_likes": int(mutual_likes),
            "social_proximity": int(social_proximity),
            "session_position": int(session_position),
            "session_genre_drift": float(session_genre_drift),
            "time_since_last_action": float(time_since_last_action),
            "is_mobile": int(is_mobile),
            "is_desktop": int(is_desktop),
            "is_tablet": int(is_tablet),
            "session_length": int(session_length),
            "global_pop_rank": float(global_pop_rank),
            "trending_score": float(trending_score),
            "intra_list_diversity": float(intra_list_diversity),
        }
        return feature_vector
    
    async def _batch_get_books_metadata(self, book_ids: List[str]) -> Dict[str, Dict]:
        """Supabase batch query"""
        if not book_ids:
            return {}
            
        response = self.supabase.table("books").select("*").in_("id", book_ids).execute()
        return {row["id"]: row for row in response.data}
  
    async def _get_ratings(self, user_id: str, book_ids: List[str]) -> Dict[str, Dict]:
        """Get ratings using PostgREST API (no .group())"""
        if not book_ids:
            return {}
        
        ratings_data = {book_id: {"user_rating": 0.0, "avg_rating": 3.5, "rating_count": 0} 
                        for book_id in book_ids}
        
        try:
            # 1. Get USER'S ratings (simple)
            user_ratings = self.supabase.table("rating") \
                .select("book_id, rating") \
                .eq("user_id", user_id) \
                .in_("book_id", book_ids) \
                .execute()
            
            for row in user_ratings.data:
                book_id = row["book_id"]
                if book_id in ratings_data:
                    ratings_data[book_id]["user_rating"] = float(row["rating"])
            
            # 2. Get ALL ratings per book → compute avg manually
            all_ratings = self.supabase.table("rating") \
                .select("book_id, rating") \
                .in_("book_id", book_ids) \
                .execute()
            
            # Manual aggregation (PostgREST can't do GROUP BY)
            book_rating_sums = {}
            book_rating_counts = {}
            
            for row in all_ratings.data:
                book_id = row["book_id"]
                rating = float(row["rating"])
                book_rating_sums[book_id] = book_rating_sums.get(book_id, 0) + rating
                book_rating_counts[book_id] = book_rating_counts.get(book_id, 0) + 1
            
            # Compute averages
            for book_id in book_ids:
                if book_id in book_rating_counts:
                    count = book_rating_counts[book_id]
                    avg_rating = book_rating_sums[book_id] / count
                    ratings_data[book_id]["avg_rating"] = round(avg_rating, 2)
                    ratings_data[book_id]["rating_count"] = count
            
            return ratings_data
            
        except Exception as e:
            logger.error(f"Rating fetch failed: {e}")
            return ratings_data


    
    async def _get_user_profile(self, user_id: str) -> Dict:
        """Supabase user profile - FIXED caching"""
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        response = self.supabase.table("user_preferences").select("*").eq("user_id", user_id).execute()
        profile = response.data[0] if response.data else {}
        self.user_cache[user_id] = profile
        return profile
    
    async def _get_current_session(self, user_id: str) -> Dict:
        """MongoDB current session info - FIXED query"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"received_at": -1}},
            {"$limit": 1},
            {"$project": {
                "event_id": 1,
                "received_at": 1,
                "event_type": 1,
                "item_id": 1
            }}
        ]
        
        doc = list(self.events_col.aggregate(pipeline))[0]
        if doc:
            now = datetime.utcnow()
            time_since = (now - doc["received_at"]).total_seconds() / 60.0  # minutes
            
            # Count session actions
            session_count = self.events_col.count_documents({
                "user_id": user_id,
                "event_id": doc["event_id"]
            })
            
            return {
                "event_id": doc["event_id"],
                "time_since_last_action": time_since,
                "session_action_count": session_count,
                "last_event": doc["event_type"],
                "last_item": doc["item_id"]
            }
        return {"time_since_last_action": 60.0, "session_action_count": 1}
    
    async def _get_recent_reads(self, user_id: str) -> List[str]:
        """MongoDB recent reads (30 days) - FIXED event types"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}
        
        pipeline = [
            {"$match": {
                "user_id": user_id,
                "event_type": {"$in": list(VALID_EVENTS)},
                "received_at": {"$gte": start_date}
            }},
            {"$group": {"_id": "$item_id"}},
            {"$limit": 50}
        ]
        return [doc["_id"] for doc in self.events_col.aggregate(pipeline)]
    
    async def _batch_social_features(self, user_id: str, book_ids: List[str]) -> Dict[str, Dict]:
        """Neo4j batch queries for social features"""
        default_social = {
            "friend_reads": 0,           # Friends who RATED this book
            "friend_avg_rating": 0.0,    # Friends' avg r.score
            "author_following": 0,       # User FOLLOWS→author→WROTE→book
            "mutual_likes": 0,           # Shared LIKES with friends
            "social_proximity": 999      # Shortest FOLLOWS path to 4★ rater
        }
        
        social_data = {bid: default_social.copy() for bid in book_ids}
        
        try:
            with self.neo4j_driver.session() as session:
                # QUERY 1: Friend reads + ratings (User→FOLLOWS→User→RATED→Book)
                q1 = """
                MATCH (u:User {id: $user_id})-[:FOLLOWS]->(f:User)-[r:RATED]->(b:Book)
                WHERE b.id IN $book_ids
                RETURN b.id, 
                    count(DISTINCT f) as friend_reads,
                    avg(r.score) as friend_avg_rating
                ORDER BY b.id
                """
                result1 = session.run(q1, user_id=user_id, book_ids=book_ids)
                for rec in result1:
                    bid = rec["b.id"]
                    social_data[bid]["friend_reads"] = int(rec["friend_reads"] or 0)
                    social_data[bid]["friend_avg_rating"] = float(rec["friend_avg_rating"] or 0.0)
                
                # QUERY 2: Author following (User→FOLLOWS→Author→WROTE→Book)
                q2 = """
                MATCH (u:User {id: $user_id})-[:FOLLOWS]->(a:Author)-[:WROTE]->(b:Book)
                WHERE b.id IN $book_ids
                RETURN b.id, count(DISTINCT a) as author_following
                ORDER BY b.id
                """
                result2 = session.run(q2, user_id=user_id, book_ids=book_ids)
                for rec in result2:
                    social_data[rec["b.id"]]["author_following"] = int(rec["author_following"] or 0)
                
                # Q3a: Mutual likes 
                q3a = """
                MATCH (u:User {id: $user_id})-[:FOLLOWS]->(f:User)-[:LIKES]->(b:Book)<-[:LIKES]-(u)
                WHERE b.id IN $book_ids
                RETURN b.id, count(DISTINCT f) as mutual_likes
                """
                result3a = session.run(q3a, user_id=user_id, book_ids=book_ids)
                for rec in result3a:
                    social_data[rec["b.id"]]["mutual_likes"] = int(rec["mutual_likes"] or 0)
                
                # Q3b: Social proximity (direct friends) 
                q3b = """
                MATCH (u:User {id: $user_id})-[:FOLLOWS]->(f:User)-[r:RATED]->(b:Book)
                WHERE b.id IN $book_ids AND r.score >= 4.0
                RETURN b.id, 1 as social_proximity
                """
                result3b = session.run(q3b, user_id=user_id, book_ids=book_ids)
                for rec in result3b:
                    social_data[rec["b.id"]]["social_proximity"] = 1
        
        except Exception as e:
            logger.error(f"Neo4j social features error: {e}")
        
        return social_data

    async def _batch_popularity_features(self, book_ids: List[str]) -> Dict[str, Dict]:
        """Redis batch pipeline"""
        if not book_ids:
            return {}
        
        pop_data = {}
        pipe = self.redis_client.pipeline()
        windows = ["multi", "7d", "30d"]
        
        # Batch ALL zscore calls
        for key_suffix in windows:
            key = f"popularity:trending:{key_suffix}"
            for book_id in book_ids:
                pipe.zscore(key, book_id)
        
        raw_scores = pipe.execute()
        
        # Parse efficiently
        for i, book_id in enumerate(book_ids):
            book_scores = {}
            idx = i * len(windows)
            for j, window in enumerate(windows):
                score = raw_scores[idx + j]
                book_scores[window] = float(score or 0)
            
            global_rank = self._score_to_rank(book_scores["multi"])
            trending_velocity = book_scores["7d"] - book_scores["30d"]
            
            pop_data[book_id] = {
                "global_rank": global_rank,
                "trending_score": trending_velocity,
                "diversity": 0.5
            }
        
        return pop_data
    
    def _score_to_rank(self, raw_score: float) -> float:
        """Score → normalized rank [0,1]"""
        return min(1.0, raw_score / 100.0) if raw_score > 0 else 0.0
    
    def _genre_drift(self, user_genres: List[str], book_genres: List[str]) -> float:
        """Cosine distance between genre sets"""
        if not user_genres or not book_genres:
            return 0.5
        
        common = len(set(user_genres) & set(book_genres))
        total = len(set(user_genres + book_genres))
        return 1.0 - (common / max(1, total))  # Distance metric

# Global instance
feature_service = FeatureService()
