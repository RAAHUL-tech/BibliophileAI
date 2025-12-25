import os
from io import BytesIO
from typing import Any, Dict
import logging
import ray
import boto3
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from neo4j import GraphDatabase
import pyarrow as pa
import pyarrow.parquet as pq


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

s3_uri = os.environ["S3_URI"]
if not s3_uri.endswith("/"):
    s3_uri += "/"
GRAPH_PREFIX = os.getenv("GRAPH_S3_PREFIX")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def _add_edge_weighted(G: nx.Graph, u: Any, v: Any, w: float):
    if G.has_edge(u, v):
        G[u][v]["weight"] += w
    else:
        G.add_edge(u, v, weight=w)


def load_graph_from_neo4j() -> nx.Graph:
    """
    Build a global user-book-author-genre graph from Neo4j.
    Adjust the query to your actual labels/relations.
    """
    query = """
    MATCH (u:User)-[r:READ|RATED]->(b:Book)
    OPTIONAL MATCH (u)-[:FOLLOWS]->(f:User)
    OPTIONAL MATCH (a:Author)-[:WROTE]->(b)
    OPTIONAL MATCH (b)-[:HAS_GENRE]->(g:Genre)
    RETURN
      id(u) AS uid,
      id(b) AS bid,
      id(f) AS fid,
      id(a) AS aid,
      id(g) AS gid,
      type(r) AS rel_type,
      coalesce(r.score, 1.0) AS rating
    """

    G = nx.Graph()

    with driver.session() as session:
        for record in session.run(query):
            uid = record["uid"]
            bid = record["bid"]
            fid = record["fid"]
            aid = record["aid"]
            gid = record["gid"]
            rel_type = record["rel_type"]
            rating = record["rating"]

            if uid is not None:
                G.add_node(uid, type="user")
            if bid is not None:
                G.add_node(bid, type="book")
            if fid is not None:
                G.add_node(fid, type="user")
            if aid is not None:
                G.add_node(aid, type="author")
            if gid is not None:
                G.add_node(gid, type="genre")

            if uid is not None and bid is not None:
                w = 1.0 if rel_type == "READ" else float(rating)
                _add_edge_weighted(G, uid, bid, w)
            if uid is not None and fid is not None:
                _add_edge_weighted(G, uid, fid, 0.5)
            if aid is not None and bid is not None:
                _add_edge_weighted(G, aid, bid, 0.3)
            if bid is not None and gid is not None:
                _add_edge_weighted(G, bid, gid, 0.2)

    return G


@ray.remote
def compute_pagerank(G_data: Dict) -> Dict[int, float]:
    G = nx.from_dict_of_dicts(G_data)
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    return pr


@ray.remote
def compute_node2vec(G_data: Dict) -> Dict[int, list]:
    G = nx.from_dict_of_dicts(G_data)
    node2vec = Node2Vec(
        G,
        dimensions=64,
        walk_length=20,
        num_walks=100,
        workers=1,
        weight_key="weight",
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {}
    for node in G.nodes():
        key = str(node)
        if key in model.wv:
            embeddings[node] = model.wv[key].tolist()
    return embeddings


def upload_parquet(df: pd.DataFrame, s3_uri: str):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    buf = BytesIO()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, buf)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logging.info(f"Uploaded Parquet → {s3_uri}")


def main():
    ray.init(address=os.getenv("RAY_ADDRESS", "local"))

    G = load_graph_from_neo4j()
    if G.number_of_nodes() == 0:
        print("Graph is empty; nothing to compute.")
        return

    G_data = nx.to_dict_of_dicts(G)

    pr_ref = compute_pagerank.remote(G_data)
    emb_ref = compute_node2vec.remote(G_data)

    pr_scores = ray.get(pr_ref)
    embeddings = ray.get(emb_ref)
    
    S3_PR_PATH = f"{s3_uri.rstrip('/')}/{GRAPH_PREFIX}/global_pagerank.parquet"
    S3_EMB_PATH = f"{s3_uri.rstrip('/')}/{GRAPH_PREFIX}/node2vec_embeddings.parquet"
    
    # PageRank scores
    pr_rows = [{"node_id": nid, "pr_score": score} for nid, score in pr_scores.items()]
    pr_df = pd.DataFrame(pr_rows)
    upload_parquet(pr_df, S3_PR_PATH)
    # Node2Vec embeddings
    emb_rows = [{"node_id": nid, "embedding": vec} for nid, vec in embeddings.items()]
    emb_df = pd.DataFrame(emb_rows)
    upload_parquet(emb_df, S3_EMB_PATH)

    print("Graph analytics completed and uploaded to S3.")


if __name__ == "__main__":
    main()
