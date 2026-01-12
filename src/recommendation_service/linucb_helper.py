import os
import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from pinecone import Pinecone
from neo4j import GraphDatabase
import redis


# Global cache (singleton) - book-only
_als_book_factors = None
_s3_client = None
_global_centroid = None
_book_centroid = None
_graph_pr_df = None
redis_client = redis.from_url(os.getenv("REDIS_URL"))


AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

def _get_s3_client():
    """Lazy S3 client"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    return _s3_client

def _load_als_book_factors():
    """Load book factors ONCE from S3 (cached)"""
    global _als_book_factors, _global_centroid
    
    if _als_book_factors is not None:
        return
    
    s3 = _get_s3_client()
    s3_uri = os.getenv("S3_URI").rstrip('/')
    als_prefix = os.getenv("ALS_S3_PREFIX", "ALS_Train")
    
    try:
        # Parse bucket from S3 URI
        bucket_name = s3_uri.split("/")[2] if "s3://" in s3_uri else s3_uri.split("/")[0]
        bucket_name = bucket_name.replace("s3://", "")
        
        # Download ONLY book factors
        book_key = f"{als_prefix}/book_factors.parquet"
        book_obj = s3.get_object(Bucket=bucket_name, Key=book_key)
        _als_book_factors = pd.read_parquet(BytesIO(book_obj["Body"].read()))
        
        # Create global book centroid (average of all book factors)
        _als_book_factors['global_similarity'] = 1.0  # Self-similarity
        book_factors_only = _als_book_factors.drop(columns=["book_id", "global_similarity"]).to_numpy()
        global_centroid = np.mean(book_factors_only, axis=0)
        
        # Cache centroid for ultra-fast scoring
        _global_centroid = global_centroid

        print(f"Loaded ALS book factors: {_als_book_factors.shape[0]} books, centroid ready")
        
    except Exception as e:
        print(f"ALS book factors load failed: {e}")
        _als_book_factors = pd.DataFrame()
        _global_centroid = np.zeros(64)  # Default 64 factors

def get_als_score(book_id: str) -> float:
    """
    Get ALS book quality score (0-1 normalized) 
    Uses global book centroid similarity (no user_id needed)
    """
    # Ensure factors are loaded
    _load_als_book_factors()
    
    if _als_book_factors.empty:
        return 0.5  # Neutral fallback
    
    # Find book vector
    book_row = _als_book_factors[_als_book_factors["book_id"] == book_id]
    if book_row.empty:
        # Unknown book → similarity to global centroid
        return 0.5
    
    book_vec = book_row.drop(columns=["book_id", "global_similarity"]).to_numpy().flatten()
    
    # Score = cosine similarity to global book centroid
    # (How "prototypical" is this book compared to all others?)
    global_sim = np.dot(book_vec, _global_centroid) / (
        np.linalg.norm(book_vec) * np.linalg.norm(_global_centroid) + 1e-8
    )
    
    # Normalize to 0-1 (sigmoid scaling)
    score = 1.0 / (1.0 + np.exp(-global_sim * 3.0))
    
    return float(np.clip(score, 0.0, 1.0))


# Singleton Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
book_index = pc.Index("book-metadata-index")


def _load_book_centroid():
    """Calculate global book centroid from Pinecone (cached)"""
    global _book_centroid
    
    if _book_centroid is not None:
        return
    
    try:
        # Fetch top 1000 popular books as centroid basis
        # Use popularity-sorted query to weight toward successful books
        dummy_query = [0.1] * 1024  # 1024-dim embeddings
        results = book_index.query(
            vector=dummy_query, 
            top_k=1000, 
            include_values=True,
            namespace="__default__"
        )
        
        # Extract all book vectors
        book_vectors = []
        for match in results.matches:
            vector_values = np.array(list(match.values)) if match.values else None
            if vector_values is not None:
                book_vectors.append(vector_values)
        
        if book_vectors:
            _book_centroid = np.mean(book_vectors, axis=0)
            print(f"Book centroid calculated from {len(book_vectors)} books")
        else:
            _book_centroid = np.zeros(1024)  # Fallback
            
    except Exception as e:
        print(f"Book centroid failed: {e}")
        _book_centroid = np.zeros(1024)

def get_content_similarity(book_id: str) -> float:
    """
    Get content similarity score for book (0-1 normalized)
    Book-to-global-centroid cosine similarity (no user_id needed)
    """
    # Ensure centroid is loaded
    _load_book_centroid()
    
    if _book_centroid is None:
        return 0.5  # Neutral fallback
    
    try:
        # Fetch specific book embedding
        query_result = book_index.fetch(ids=[str(book_id)], namespace="__default__")
        book_vectors = query_result.vectors
        book_record = book_vectors.get(str(book_id), None)
        
        if not book_record or not book_record.values:
            return 0.5  # Unknown book
        
        # Extract book vector
        book_vec = np.array(list(book_record.values))
        
        # Cosine similarity to global book centroid
        dot_product = np.dot(book_vec, _book_centroid)
        norm_product = np.linalg.norm(book_vec) * np.linalg.norm(_book_centroid)
        cosine_sim = dot_product / (norm_product + 1e-8)
        
        # Normalize to 0-1 range
        score = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))
        
        return score
        
    except Exception as e:
        print(f"Content similarity failed for {book_id}: {e}")
        return 0.5


def _load_graph_pagerank():
    """Load global_pagerank.parquet from S3 (cached)"""
    global _graph_pr_df
    
    if _graph_pr_df is not None:
        return
    
    s3_uri = os.getenv("S3_URI").rstrip('/')
    graph_prefix = os.getenv("GRAPH_S3_PREFIX", "Graph_Train")
    
    try:
        bucket_name = s3_uri.split("/")[2] if "s3://" in s3_uri else s3_uri.split("/")[0]
        pr_key = f"{graph_prefix}/global_pagerank.parquet"
        
        pr_obj = _get_s3_client().get_object(Bucket=bucket_name, Key=pr_key)
        _graph_pr_df = pd.read_parquet(BytesIO(pr_obj["Body"].read()))
        
        # Normalize to 0-1 range
        min_pr, max_pr = _graph_pr_df['pr_score'].min(), _graph_pr_df['pr_score'].max()
        if max_pr > min_pr:
            _graph_pr_df['pr_normalized'] = (_graph_pr_df['pr_score'] - min_pr) / (max_pr - min_pr)
        else:
            _graph_pr_df['pr_normalized'] = 0.5
            
        print(f"Loaded {_graph_pr_df.shape[0]} PageRank scores")
        
    except Exception as e:
        print(f"PageRank parquet failed: {e}")
        _graph_pr_df = pd.DataFrame({'node_id': [], 'pr_normalized': []})

def get_graph_pagerank(book_id: str) -> float:
    """
    Get normalized PageRank score for book (0-1)
    S3 parquet → Redis cache → Neo4j fallback
    """
    # 1. Ultra-fast Redis cache (5min TTL)
    cache_key = f"graph:pr:{book_id}"
    cached_pr = redis_client.get(cache_key)
    if cached_pr:
        return float(cached_pr)
    
    # 2. Load S3 parquet (cached globally)
    _load_graph_pagerank()
    
    # 3. Lookup in DataFrame
    book_row = _graph_pr_df[_graph_pr_df['node_id'] == book_id]
    if not book_row.empty:
        pr_score = float(book_row['pr_normalized'].iloc[0])
        # Cache in Redis
        redis_client.setex(cache_key, 300, pr_score)  # 5min TTL
        return pr_score
    
    # 4. Neo4j fallback (live query)
    pr_score = _neo4j_pagerank_fallback(book_id)
    
    # Cache fallback result
    redis_client.setex(cache_key, 300, pr_score)
    return pr_score

def _neo4j_pagerank_fallback(book_id: str) -> float:
    """Live Neo4j query if not in S3 parquet"""
    
    neo4j_uri = os.getenv("NEO4J_URI")
    driver = GraphDatabase.driver(neo4j_uri, 
                                 auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (b:Book {id: $book_id})-[:READ*1..2]-(u:User)
                RETURN count(DISTINCT u) * 1.0 / 1000 as live_pr
            """, book_id=book_id)
            record = result.single()
            if record and record["live_pr"]:
                raw_pr = float(record["live_pr"])
                # Normalize (assuming 0-10 range from your training)
                normalized = min(raw_pr / 10.0, 1.0)
                return normalized
    except Exception as e:
        print(f"Neo4j fallback failed for {book_id}: {e}")
    finally:
        driver.close()
    
    return 0.1  # Minimum PageRank
