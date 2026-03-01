"""
LinUCB: online contextual bandit for personalized book ranking.
- One linear model per user: θ, A, b stored in Redis. score = θ^T x + α√(x^T A^{-1} x).
- Feature x = [x_user, x_book, x_interaction]: user preferences, book metadata, genre/author match, cosine_sim.
- On refresh-on-logout: reward from MongoDB (read=3, page_turn=2, bookmark=3, review=4), update A, b, θ.
"""
import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import redis
import httpx
from pymongo import MongoClient
from datetime import datetime, timedelta
from pinecone import Pinecone
from linucb_helper import get_content_similarity, get_als_score, get_graph_pagerank

logger = logging.getLogger(__name__)

# Reward weights for session events (MongoDB)
REWARD_READ = 3.0
REWARD_PAGE_TURN = 2.0
REWARD_BOOKMARK = 3.0
REWARD_REVIEW = 4.0
REWARD_MAX = 4.0  # normalize to [0, 1] by dividing by this

VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}
FEATURE_DIM = 16
LINUCB_ALPHA = 1.0
LINUCB_LAMBDA = 1.0  # regularization for cold start A = λI

REDIS_KEY_MODEL = "linucb:model:{}"
REDIS_KEY_SHOWN = "linucb:shown:{}"
MODEL_TTL = 365 * 24 * 3600  # 1 year
SHOWN_TTL = 7 * 24 * 3600   # 7 days

SUPABASE_URL =  os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

def _supabase_headers() -> Dict[str, str]:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _fetch_user_profile_sync(user_id: str) -> Dict[str, Any]:
    """Sync fetch user_preferences from Supabase (genres, authors, preferred_language)."""
    url = f"{SUPABASE_URL}/rest/v1/user_preferences"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                url,
                headers=headers,
                params={"user_id": f"eq.{user_id}", "select": "genres,authors,preferred_language"},
            )
            r.raise_for_status()
            rows = r.json()
            return rows[0] if rows else {}
    except Exception as e:
        logger.warning("LinUCB: user profile fetch failed for %s: %s", user_id, e)
        return {}


def _fetch_books_metadata_sync(book_ids: List[str]) -> Dict[str, Dict]:
    """Sync fetch books metadata from Supabase (authors, categories)."""
    if not book_ids:
        return {}
    idlist = ",".join([f'"{bid}"' for bid in book_ids])
    url = f"{SUPABASE_URL}/rest/v1/books?select=id,authors,categories&id=in.({idlist})"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url,  headers=headers)
            r.raise_for_status()
            rows = r.json()
            return {str(row["id"]): row for row in rows}
    except Exception as e:
        logger.warning("LinUCB: books metadata fetch failed: %s", e)
        return {}


def _cosine_sim_user_book_sync(user_id: str, book_id: str) -> float:
    """Cosine similarity between user embedding and book embedding (Pinecone). 0-1."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        user_index = pc.Index("user-preferences-index")
        book_index = pc.Index("book-metadata-index")
        u = user_index.fetch(ids=[str(user_id)], namespace="__default__")
        b = book_index.fetch(ids=[str(book_id)], namespace="__default__")
        uv = u.vectors.get(str(user_id))
        bv = b.vectors.get(str(book_id))
        if not uv or not uv.values or not bv or not bv.values:
            return 0.5
        a = np.array(uv.values)
        b_vec = np.array(bv.values)
        dot = np.dot(a, b_vec)
        n = np.linalg.norm(a) * np.linalg.norm(b_vec) + 1e-8
        sim = dot / n
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
    except Exception as e:
        logger.debug("LinUCB: cosine_sim failed %s/%s: %s", user_id, book_id, e)
        return 0.5


def _build_feature_vector(
    user_id: str,
    book_id: str,
    user_profile: Dict[str, Any],
    book_meta: Dict[str, Any],
) -> np.ndarray:
    """
    x = [x_user, x_book, x_interaction]. Fixed dim FEATURE_DIM.
    - User: preferred genres (top 3 as 0/1), preferred_language match, placeholder.
    - Book: categories (top 3), authors present, popularity.
    - Interaction: genre_match, author_match, cosine_sim(user_emb, book_emb).
    """
    user_genres = user_profile.get("genres") or []
    user_authors = user_profile.get("authors") or []
    if not isinstance(user_genres, list):
        user_genres = list(user_genres) if user_genres else []
    if not isinstance(user_authors, list):
        user_authors = list(user_authors) if user_authors else []
    book_cats = book_meta.get("categories") or []
    book_authors = book_meta.get("authors") or []
    if not isinstance(book_cats, list):
        book_cats = list(book_cats) if book_cats else []
    if not isinstance(book_authors, list):
        book_authors = list(book_authors) if book_authors else []

    genre_match = 1.0 if (set(user_genres) & set(book_cats)) else 0.0
    author_match = 1.0 if (set(user_authors) & set(book_authors)) else 0.0
    cosine_sim = _cosine_sim_user_book_sync(user_id, book_id)

    # User features (5): bias=1, top 3 genre indicators, lang match placeholder
    u1 = 1.0 if user_genres and book_cats and (user_genres[0] in book_cats) else 0.0
    u2 = 1.0 if len(user_genres) > 1 and book_cats and (user_genres[1] in book_cats) else 0.0
    u3 = 1.0 if len(user_genres) > 2 and book_cats and (user_genres[2] in book_cats) else 0.0
    user_part = [1.0, u1, u2, u3, 0.5]

    # Book features (4): top 2 category indicators, author count norm, popularity
    b1 = 1.0 if book_cats else 0.0
    b2 = 1.0 if len(book_cats) > 1 else 0.0
    als = get_als_score(book_id)
    content = get_content_similarity(book_id)
    book_part = [b1, b2, als, content]

    # Interaction (3) + pad to 16
    interaction = [genre_match, author_match, cosine_sim]
    graph_pr = get_graph_pagerank(book_id)
    pop = 0.5  # could read from Redis popularity
    x = np.array(user_part + book_part + interaction + [graph_pr, pop, 0.0, 0.0], dtype=np.float32)
    if len(x) > FEATURE_DIM:
        x = x[:FEATURE_DIM]
    elif len(x) < FEATURE_DIM:
        x = np.pad(x, (0, FEATURE_DIM - len(x)), constant_values=0.0)
    print(f"LinUCB: feature vector for user {user_id} and book {book_id}: {x}")
    return x.reshape(-1, 1)  # column vector (d, 1)


class LinUCBServe:
    def __init__(self):
        self.redis_client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        self.mongo_client = MongoClient(os.environ.get("MONGO_URI", ""))
        self.alpha = float(os.environ.get("LINUCB_ALPHA", LINUCB_ALPHA))
        self.lambda_ = float(os.environ.get("LINUCB_LAMBDA", LINUCB_LAMBDA))

    def _load_model(self, user_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load (A, b) for user from Redis. Cold start: A = λI, b = 0."""
        key = REDIS_KEY_MODEL.format(user_id)
        try:
            raw = self.redis_client.get(key)
            if raw:
                data = json.loads(raw)
                A = np.array(data["A"], dtype=np.float64)
                b = np.array(data["b"], dtype=np.float64).reshape(-1, 1)
                return A, b
        except Exception as e:
            logger.warning("LinUCB: load model failed for %s: %s", user_id, e)
        # Cold start
        A = self.lambda_ * np.eye(FEATURE_DIM, dtype=np.float64)
        b = np.zeros((FEATURE_DIM, 1), dtype=np.float64)
        return A, b

    def _save_model(self, user_id: str, A: np.ndarray, b: np.ndarray) -> None:
        key = REDIS_KEY_MODEL.format(user_id)
        try:
            payload = json.dumps({"A": A.tolist(), "b": b.ravel().tolist()})
            self.redis_client.setex(key, MODEL_TTL, payload)
        except Exception as e:
            logger.warning("LinUCB: save model failed for %s: %s", user_id, e)

    def update(self, user_id: str, x: np.ndarray, reward: float) -> None:
        """Online update: A += x x^T, b += reward * x; then persist."""
        A, b = self._load_model(user_id)
        x = x.reshape(-1, 1)
        A = A + x @ x.T
        b = b + reward * x
        self._save_model(user_id, A, b)

    def score(self, user_id: str, x: np.ndarray) -> float:
        """p = θ^T x + α * sqrt(x^T A^{-1} x), with θ = A^{-1} b."""
        A, b = self._load_model(user_id)
        x = x.reshape(-1, 1)
        try:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            mean = (theta.T @ x).item()
            unc = (np.sqrt(x.T @ A_inv @ x)).item()
            return mean + self.alpha * unc
        except np.linalg.LinAlgError:
            return 0.5

    def set_shown_books(self, user_id: str, book_ids: List[str]) -> None:
        key = REDIS_KEY_SHOWN.format(user_id)
        try:
            self.redis_client.setex(key, SHOWN_TTL, json.dumps(book_ids))
        except Exception as e:
            logger.warning("LinUCB: set shown failed: %s", e)

    def get_shown_books(self, user_id: str) -> List[str]:
        key = REDIS_KEY_SHOWN.format(user_id)
        try:
            raw = self.redis_client.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return []

    def get_rewards_for_books(self, user_id: str, book_ids: List[str]) -> Dict[str, float]:
        """
        From MongoDB: last session events for this user. For each book_id in book_ids,
        sum event rewards (read=3, page_turn=2, bookmark=3, review=4), normalize by REWARD_MAX.
        If user did not interact with the book, reward = 0.
        """
        if not book_ids:
            return {}
        col = self.mongo_client["click_stream"]["events"]
        # Last 30 days, then take last session (consecutive events)
        end = datetime.utcnow()
        start = end - timedelta(days=30)
        pipeline = [
            {"$match": {
                "user_id": user_id,
                "item_id": {"$in": book_ids},
                "event_type": {"$in": list(VALID_EVENTS)},
                "received_at": {"$gte": start, "$lte": end},
            }},
            {"$sort": {"received_at": 1}},
            {"$project": {"item_id": 1, "event_type": 1, "received_at": 1, "metadata": 1}},
        ]
        events = list(col.aggregate(pipeline))
        # Aggregate per book: max reward (or sum then normalize)
        book_reward_raw: Dict[str, float] = {bid: 0.0 for bid in book_ids}
        for e in events:
            bid = str(e.get("item_id", ""))
            if bid not in book_reward_raw:
                continue
            t = e.get("event_type")
            meta = e.get("metadata") or {}
            if t == "review":
                r = meta.get("rating", 0)
                book_reward_raw[bid] = max(book_reward_raw[bid], REWARD_REVIEW if r >= 4 else REWARD_REVIEW - 1)
            elif t == "read":
                book_reward_raw[bid] = max(book_reward_raw[bid], REWARD_READ)
            elif t == "page_turn":
                book_reward_raw[bid] = max(book_reward_raw[bid], REWARD_PAGE_TURN)
            elif t == "bookmark_add":
                book_reward_raw[bid] = max(book_reward_raw[bid], REWARD_BOOKMARK)
        # Normalize to [0, 1]
        return {bid: min(1.0, r / REWARD_MAX) for bid, r in book_reward_raw.items()}

    def train_on_logout(self, user_id: str) -> None:
        """
        Online training step on logout: get previously shown LinUCB books,
        compute rewards from MongoDB, update (A, b) for each (user, book, reward).
        """
        shown = self.get_shown_books(user_id)
        if not shown:
            return
        rewards = self.get_rewards_for_books(user_id, shown)
        user_profile = _fetch_user_profile_sync(user_id)
        books_meta = _fetch_books_metadata_sync(shown)
        for book_id in shown:
            r = rewards.get(book_id, 0.0)
            book_meta = books_meta.get(book_id, {})
            x = _build_feature_vector(user_id, book_id, user_profile, book_meta)
            self.update(user_id, x, r)
        self.set_shown_books(user_id, [])  # clear so next time we only train on new shown

    def get_linucb_ranked(self, book_ids: List[str], user_id: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """
        Score all candidate books with LinUCB: p = θ^T x + α√(x^T A^{-1} x).
        Returns (book_ids, scores) sorted by score descending.
        Caller should call set_shown_books(user_id, returned_book_ids) after.
        """
        if not book_ids:
            return [], []
        if not user_id:
            return book_ids[:200], [0.5] * min(200, len(book_ids))
        user_profile = _fetch_user_profile_sync(user_id)
        books_meta = _fetch_books_metadata_sync(book_ids)
        scored = []
        for book_id in book_ids:
            book_meta = books_meta.get(book_id, {})
            x = _build_feature_vector(user_id, book_id, user_profile, book_meta)
            s = self.score(user_id, x)
            scored.append((book_id, float(s)))
        scored.sort(key=lambda t: t[1], reverse=True)
        out_ids = [t[0] for t in scored[:200]]
        out_scores = [t[1] for t in scored[:200]]
        return out_ids, out_scores


linucb_ranker = LinUCBServe()
