from neo4j import GraphDatabase
from typing import List, Tuple, Dict
import os

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

ALPHA = 0.85        # damping
MAX_ITERS = 30
EPS = 1e-6

def _build_graph(tx, user_id: str):
    """
    Build a compact graph (nodes + weighted edges) around the user.
    Returns: (node_ids, edges, seed_weights, friends_for_social_boost)
    - node_ids: list of node ids (Neo4j internal ids or custom ids)
    - edges: list of (src_idx, dst_idx, weight)
    - seed_weights: dict[node_idx] -> weight (for seed vector s)
    - friends_like_book: dict[book_idx] -> count of friends who rated >= 4
    """
    # Fetch nodes and edges relevant to the user
    query = """
    MATCH (u:User {id: $user_id})

    // candidate books: neighbors within 2 hops or interacted by friends
    MATCH (u)-[:FOLLOWS|READ|RATED]->(x)
    OPTIONAL MATCH (u)-[:FOLLOWS]->(f:User)-[:RATED]->(b:Book)
    OPTIONAL MATCH (u)-[:READ|RATED]->(b2:Book)
    WITH collect(DISTINCT u)      AS users,
         collect(DISTINCT f)      AS friends,
         collect(DISTINCT b)      AS friendBooks,
         collect(DISTINCT b2)     AS selfBooks

    // all candidate books
    WITH users, friends, selfBooks + friendBooks AS books

    // bring authors and genres too
    OPTIONAL MATCH (a:Author)-[:WROTE]->(b:Book) WHERE b IN books
    OPTIONAL MATCH (b)-[:HAS_GENRE]->(g:Genre) WHERE b IN books

    WITH
      apoc.coll.toSet(users)   AS users,
      apoc.coll.toSet(friends) AS friends,
      apoc.coll.toSet(books)   AS books,
      collect(DISTINCT a)      AS authors,
      collect(DISTINCT g)      AS genres

    WITH users + friends AS userNodes, books AS bookNodes,
         authors AS authorNodes, genres AS genreNodes

    WITH apoc.coll.toSet(userNodes + bookNodes + authorNodes + genreNodes) AS nodes

    // collect all weighted relationships among these nodes
    UNWIND nodes AS n
    OPTIONAL MATCH (n)-[r:FOLLOWS|READ|RATED|WROTE|HAS_GENRE]->(m)
    WHERE m IN nodes
    RETURN
      collect(DISTINCT id(n)) AS nodeIds,
      collect(DISTINCT [id(startNode(r)), id(endNode(r)), 
           CASE type(r)
             WHEN 'RATED' THEN coalesce(r.score, 1.0)
             WHEN 'READ'  THEN 1.0
             WHEN 'FOLLOWS' THEN 0.5
             WHEN 'WROTE' THEN 0.3
             WHEN 'HAS_GENRE' THEN 0.2
             ELSE 0.1
           END]) AS edges;
    """

    result = tx.run(query, user_id=user_id).single()
    if not result:
        return [], [], {}, {}

    node_ids: List[int] = result["nodeIds"]
    raw_edges = result["edges"]

    # Map Neo4j node id -> index (0..N-1)
    idx_map: Dict[int, int] = {nid: i for i, nid in enumerate(node_ids)}

    edges: List[Tuple[int, int, float]] = []
    for src_nid, dst_nid, w in raw_edges:
        if src_nid in idx_map and dst_nid in idx_map:
            edges.append((idx_map[src_nid], idx_map[dst_nid], float(w)))

    # Seed vector: user + their directly rated books
    seed_weights: Dict[int, float] = {}
    seed_query = """
    MATCH (u:User {id: $user_id})
    OPTIONAL MATCH (u)-[r:RATED]->(b:Book)
    RETURN id(u) AS uid, collect([id(b), coalesce(r.score, 1.0)]) AS ratedBooks
    """
    seed_res = tx.run(seed_query, user_id=user_id).single()
    if seed_res:
        uid = seed_res["uid"]
        rated = seed_res["ratedBooks"]
        if uid in idx_map:
            seed_weights[idx_map[uid]] = 1.0
        for nid, score in rated:
            if nid in idx_map:
                seed_weights[idx_map[nid]] = seed_weights.get(idx_map[nid], 0.0) + float(score)

    # Normalize seed weights
    total_w = sum(seed_weights.values())
    if total_w > 0:
        for k in list(seed_weights.keys()):
            seed_weights[k] /= total_w

    # SocialBoost: friends who liked each book (score >=4)
    friends_query = """
    MATCH (u:User {id: $user_id})-[:FOLLOWS]->(f:User)-[r:RATED]->(b:Book)
    WHERE r.score >= 4
    RETURN id(f) AS fid, id(b) AS bid
    """
    friends_res = tx.run(friends_query, user_id=user_id)
    friend_likes: Dict[int, set] = {}
    friends_set: set = set()
    for row in friends_res:
        fid = row["fid"]
        bid = row["bid"]
        friends_set.add(fid)
        if bid not in friend_likes:
            friend_likes[bid] = set()
        friend_likes[bid].add(fid)

    # convert to index-based
    friends_for_social_boost: Dict[int, float] = {}
    num_friends = len(friends_set) or 1
    for book_nid, fset in friend_likes.items():
        if book_nid in idx_map:
            frac = len(fset) / num_friends
            friends_for_social_boost[idx_map[book_nid]] = frac

    return node_ids, edges, seed_weights, friends_for_social_boost


def _personalized_pagerank(node_ids, edges, seed_weights, friends_boost,
                           alpha=ALPHA, max_iters=MAX_ITERS, eps=EPS):
    """
    Compute r using r(t+1) = (1-alpha)*s + alpha * P^T r(t)
    edges: list (src_idx, dst_idx, weight)
    """
    n = len(node_ids)
    if n == 0:
        return []

    # Build transition matrix P (column-stochastic) as adjacency lists
    out_weight = [0.0] * n
    incoming = [[] for _ in range(n)]  # incoming[j] = list of (i, weight_ij)

    for src, dst, w in edges:
        out_weight[src] += w
        incoming[dst].append((src, w))

    # Normalize to make each column sum to 1 (P_ij as defined in text)
    for j in range(n):
        total = sum(w for (i, w) in incoming[j])
        if total > 0:
            incoming[j] = [(i, w / total) for (i, w) in incoming[j]]

    # Seed vector s
    s = [0.0] * n
    if seed_weights:
        for idx, w in seed_weights.items():
            s[idx] = w
    else:
        # fallback: uniform
        for i in range(n):
            s[i] = 1.0 / n

    r = s[:]  # initial ranking

    for _ in range(max_iters):
        new_r = [0.0] * n
        for j in range(n):
            contrib = 0.0
            for (i, pij) in incoming[j]:
                contrib += pij * r[i]
            new_r[j] = (1 - alpha) * s[j] + alpha * contrib

        diff = sum(abs(new_r[i] - r[i]) for i in range(n))
        r = new_r
        if diff < eps:
            break

    # Post-process with SocialBoost
    beta = 0.2
    scored = []
    for idx, base in enumerate(r):
        boost = friends_boost.get(idx, 0.0)
        scored.append(base * (1 + beta * boost))
    return scored


def graph_recommend_books(user_id: str, top_k: int = 50):
    """
    Main entry: returns list of (book_id, score) for user.
    """
    with driver.session() as session:
        node_ids, edges, seed_weights, friends_boost = session.execute_read(_build_graph, user_id)

    scores = _personalized_pagerank(node_ids, edges, seed_weights, friends_boost)
    print(f"Graph PageRank scores for user {user_id}: {scores}")
    # Filter only Book nodes
    def get_books(tx):
        q = """
        UNWIND $ids AS nid
        MATCH (n) WHERE id(n) = nid
        WITH n, labels(n) AS lbls
        WHERE 'Book' IN lbls
        RETURN id(n) AS nid, n.id AS bookId  // 'id' is domain id for book
        """
        res = tx.run(q, ids=node_ids)
        return {row["nid"]: row["bookId"] for row in res}

    with driver.session() as session:
        nid_to_book_id = session.execute_read(get_books)

    book_scores = []
    for idx, nid in enumerate(node_ids):
        if nid in nid_to_book_id:
            book_scores.append((nid_to_book_id[nid], scores[idx]))

    # sort by score desc, take top_k
    book_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"3. Graph Recommendations for user {user_id}: {book_scores[:top_k]}")
    return book_scores[:top_k]
