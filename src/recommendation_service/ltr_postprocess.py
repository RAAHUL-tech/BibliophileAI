"""
Stage 5: Post-Processing & Diversity Control for LTR (Top Picks) recommendations.
Applies author/genre diversity constraints, MMR for diversity-relevance balance,
and preferred-language business rule. Only applied to LTR output.
"""
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from pinecone import Pinecone
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Diversity constraints 
M_AUTHOR = 5  # max books per author in final list
G_MIN = 3     # minimum distinct genres in final list (3-5)
LAMBDA_MMR = 0.7  # relevance vs diversity trade-off (70% relevance, 30% diversity)
TARGET_K = 20     # target size of final list

BOOK_INDEX_NAME = "book-metadata-index"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
book_index = pc.Index(BOOK_INDEX_NAME)

def _get_pinecone_book_vectors(book_ids: List[str]) -> Dict[str, np.ndarray]:
    """Fetch book embeddings from Pinecone (sync). Returns dict book_id -> vector (1024-dim)."""
    if not book_ids:
        return {}
    try:
        fetch_result = book_index.fetch(
            ids=[str(bid) for bid in book_ids],
            namespace="__default__"
        )
        vectors = getattr(fetch_result, "vectors", None) or {}
        out = {}
        for bid in book_ids:
            sid = str(bid)
            rec = vectors.get(sid)
            if rec and getattr(rec, "values", None):
                out[bid] = np.array(rec.values, dtype=np.float64)
        return out
    except Exception as e:
        logger.warning("LTR postprocess: Pinecone fetch failed: %s", e)
        return {}


def _normalize_scores(scores: List[float]) -> np.ndarray:
    """Normalize scores to [0, 1]. If all same or single element, return 1.0."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    smin, smax = arr.min(), arr.max()
    if smax - smin <= 1e-9:
        return np.ones_like(arr)
    return (arr - smin) / (smax - smin)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity in [0, 1] (assuming non-negative embeddings)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    sim = np.dot(v1, v2) / (n1 * n2)
    return float(np.clip(sim, 0.0, 1.0))


def _authors_for_book(book_meta: Dict[str, Any]) -> List[str]:
    """Extract list of authors from book meta."""
    a = book_meta.get("authors")
    if a is None:
        return []
    if isinstance(a, list):
        return [str(x).strip() for x in a if x]
    return [str(a).strip()] if a else []


def _genres_for_book(book_meta: Dict[str, Any]) -> List[str]:
    """Extract list of genres (categories) from book meta."""
    g = book_meta.get("categories")
    if g is None:
        return []
    if isinstance(g, list):
        return [str(x).strip() for x in g if x]
    return [str(g).strip()] if g else []


def _apply_author_diversity(
    book_ids: List[str],
    scores: List[float],
    books_meta: Dict[str, Dict[str, Any]],
    max_per_author: int = M_AUTHOR,
) -> Tuple[List[str], List[float]]:
    """Limit to at most max_per_author books per author. Keep order by score."""
    author_count: Dict[str, int] = {}
    out_ids: List[str] = []
    out_scores: List[float] = []
    for bid, sc in zip(book_ids, scores):
        meta = books_meta.get(bid) or {}
        authors = _authors_for_book(meta)
        if not authors:
            out_ids.append(bid)
            out_scores.append(sc)
            continue
        # Count by first author (or any author) - spec says limit per author
        ok = True
        for a in authors:
            if author_count.get(a, 0) >= max_per_author:
                ok = False
                break
        if not ok:
            continue
        out_ids.append(bid)
        out_scores.append(sc)
        for a in authors:
            author_count[a] = author_count.get(a, 0) + 1
    return out_ids, out_scores


def _mmr_select(
    ranked_ids: List[str],
    ranked_scores: List[float],
    embeddings_map: Dict[str, np.ndarray],
    lambda_: float = LAMBDA_MMR,
    K: int = TARGET_K,
) -> Tuple[List[str], List[float]]:
    """
    Maximal Marginal Relevance: iteratively select books maximizing
    lambda_ * Relevance(bi) - (1 - lambda_) * max_{bj in S} Similarity(bi, bj).
    Relevance = normalized LTR score in [0,1].
    """
    if not ranked_ids or K <= 0:
        return [], []
    relevance = _normalize_scores(ranked_scores)
    rel_map = {bid: relevance[i] for i, bid in enumerate(ranked_ids)}
    R = list(ranked_ids)
    S: List[str] = []
    S_scores: List[float] = []
    while len(S) < K and R:
        best_bid = None
        best_score = -1e9
        for bi in R:
            rel = rel_map.get(bi, 0.0)
            mmr = lambda_ * rel
            if S:
                max_sim = 0.0
                v_i = embeddings_map.get(bi)
                if v_i is not None:
                    for bj in S:
                        v_j = embeddings_map.get(bj)
                        if v_j is not None:
                            sim = _cosine_similarity(v_i, v_j)
                            max_sim = max(max_sim, sim)
                mmr -= (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_bid = bi
        if best_bid is None:
            break
        S.append(best_bid)
        idx = R.index(best_bid)
        S_scores.append(ranked_scores[idx])
        R.remove(best_bid)
    return S, S_scores


def _ensure_genre_diversity(
    selected_ids: List[str],
    selected_scores: List[float],
    candidate_ids: List[str],
    candidate_scores: List[float],
    books_meta: Dict[str, Dict[str, Any]],
    min_genres: int = G_MIN,
) -> Tuple[List[str], List[float]]:
    """If distinct genres in selected < min_genres, try to swap in books that add new genres."""
    def genres_in(ids: List[str]) -> set:
        gs = set()
        for bid in ids:
            gs.update(_genres_for_book(books_meta.get(bid) or {}))
        return gs

    current_genres = genres_in(selected_ids)
    if len(current_genres) >= min_genres:
        return selected_ids, selected_scores

    remaining = [(bid, sc) for bid, sc in zip(candidate_ids, candidate_scores) if bid not in selected_ids]
    out_ids = list(selected_ids)
    out_scores = list(selected_scores)
    for bid, sc in remaining:
        if len(current_genres) >= min_genres:
            break
        new_gs = _genres_for_book(books_meta.get(bid) or {})
        if not new_gs:
            continue
        added = set(new_gs) - current_genres
        if not added:
            continue
        # Swap: remove the lowest-score item that doesn't reduce genre count below current - 1
        # Simple: remove last (lowest relevance) and add this one
        out_ids.pop()
        out_scores.pop()
        out_ids.append(bid)
        out_scores.append(sc)
        current_genres.update(new_gs)
    return out_ids, out_scores


def _apply_language_filter(
    selected_ids: List[str],
    selected_scores: List[float],
    books_meta: Dict[str, Dict[str, Any]],
    user_preferred_languages: Optional[List[str]],
) -> Tuple[List[str], List[float]]:
    """
    Business rule: only include books in the user's preferred languages (if provided).
    If user_preferred_languages is None or empty, all books are kept.
    user_preferred_languages can be a list (e.g. ["en"]) or a single string (e.g. "en").
    """
    if not user_preferred_languages:
        return list(selected_ids), list(selected_scores)

    # Normalize to list so "en" -> ["en"] (avoid iterating string as ["e","n"])
    if isinstance(user_preferred_languages, str):
        preferred_list = [user_preferred_languages.lower()]
    else:
        preferred_list = [str(x).lower() for x in user_preferred_languages if x]

    def language_ok(meta: Dict) -> bool:
        lang = meta.get("language")
        if lang is None:
            return True
        return str(lang).lower() in preferred_list

    out_ids = []
    out_scores = []
    for bid, sc in zip(selected_ids, selected_scores):
        if language_ok(books_meta.get(bid) or {}):
            out_ids.append(bid)
            out_scores.append(sc)
    return out_ids, out_scores


def _compute_ild(selected_ids: List[str], embeddings_map: Dict[str, np.ndarray]) -> float:
    """Intra-List Diversity: average pairwise (1 - similarity). Higher = more diverse."""
    n = len(selected_ids)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            vi = embeddings_map.get(selected_ids[i])
            vj = embeddings_map.get(selected_ids[j])
            if vi is not None and vj is not None:
                sim = _cosine_similarity(vi, vj)
                total += 1.0 - sim
                count += 1
    if count == 0:
        return 0.0
    return total / count


def _compute_genre_entropy(selected_ids: List[str], books_meta: Dict[str, Dict[str, Any]]) -> float:
    """Genre entropy H(S) = - sum_g p_g log2(p_g)."""
    genres: List[str] = []
    for bid in selected_ids:
        genres.extend(_genres_for_book(books_meta.get(bid) or {}))
    if not genres:
        return 0.0
    n = len(genres)
    cnt = Counter(genres)
    H = 0.0
    for g, c in cnt.items():
        p = c / n
        if p > 0:
            H -= p * (np.log2(p) if p > 0 else 0)
    return float(H)


def apply_ltr_postprocess(
    book_ids: List[str],
    scores: List[float],
    books_meta: Dict[str, Dict[str, Any]],
    user_id: str,
    user_preferred_languages: Optional[List[str]] = None,
    ratings_map: Optional[Dict[str, float]] = None,
    target_k: int = TARGET_K,
    lambda_mmr: float = LAMBDA_MMR,
    m_author: int = M_AUTHOR,
    g_min: int = G_MIN,
    apply_business_rules_flag: bool = True,
) -> Tuple[List[str], List[float], Dict[str, float]]:
    """
    Full post-processing pipeline for LTR (Top Picks) output.
    Steps: author diversity cap -> MMR selection -> genre diversity -> language filter.
    Returns (final_ids, final_scores, metrics_dict with ild, genre_entropy).
    """
    if not book_ids or not scores:
        return [], [], {"ild": 0.0, "genre_entropy": 0.0}

    ratings_map = ratings_map or {}
    # Ensure we have books_meta entries for all ids
    for bid in book_ids:
        if bid not in books_meta:
            books_meta[bid] = {}

    # 1. Author diversity: cap candidates so we don't over-select one author
    cand_ids, cand_scores = _apply_author_diversity(
        book_ids, scores, books_meta, max_per_author=m_author
    )
    
    if not cand_ids:
        return [], [], {"ild": 0.0, "genre_entropy": 0.0}

    # 2. Fetch embeddings for MMR
    all_ids = list(dict.fromkeys(cand_ids + book_ids))
    embeddings_map = _get_pinecone_book_vectors(all_ids)
    if not embeddings_map:
        # No embeddings: fall back to relevance-only order (no MMR)
        selected_ids = cand_ids[:target_k]
        selected_scores = cand_scores[:target_k]
    else:
        # 3. MMR selection
        selected_ids, selected_scores = _mmr_select(
            cand_ids, cand_scores, embeddings_map, lambda_=lambda_mmr, K=target_k
        )
        
        # 4. Genre diversity
        selected_ids, selected_scores = _ensure_genre_diversity(
            selected_ids, selected_scores, cand_ids, cand_scores, books_meta, min_genres=g_min
        )
       
    # 5. Business rule: language filter (preferred languages only)
    if apply_business_rules_flag:
        selected_ids, selected_scores = _apply_language_filter(
            selected_ids, selected_scores, books_meta, user_preferred_languages
        )

    # Cap to target_k
    selected_ids = selected_ids[:target_k]
    selected_scores = selected_scores[:target_k]
    # Diversity metrics
    ild = _compute_ild(selected_ids, embeddings_map)
    genre_entropy = _compute_genre_entropy(selected_ids, books_meta)
    metrics = {"ild": round(ild, 4), "genre_entropy": round(genre_entropy, 4)}
    logger.info("LTR postprocess: %d books, ILD=%.4f, genre_entropy=%.4f", len(selected_ids), ild, genre_entropy)
    return selected_ids, selected_scores, metrics
