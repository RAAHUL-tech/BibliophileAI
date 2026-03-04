# Graph Training (Node2Vec)

Builds a unified graph from Neo4j (users, books, authors, genres, and relations such as READ, RATED, FOLLOWS, WROTE, HAS_GENRE), runs Node2Vec to learn node embeddings, and uploads embeddings to S3 for graph-based recommendation.

## Algorithm

- **Graph**: Heterogeneous graph with weighted edges. User–book edges from READ/RATED (weight from rating); user–user from FOLLOWS; book–author (WROTE), book–genre (HAS_GENRE). All nodes (user, book, author, genre) get an embedding.
- **Node2Vec**: Random-walk based embedding. Produces a fixed-size vector per node so that nearby nodes in the graph have similar vectors. Used for “social” / graph-based recommendations: e.g. recommend books close in embedding space to the user’s neighborhood.
- **Library**: `networkx` for graph construction, `node2vec.Node2Vec` for walks and embedding.

## Implementation in this project

- **graph_train.py**:
  1. **Load**: Neo4j query fetches User–READ/RATED→Book, User–FOLLOWS→User, Author–WROTE→Book, Book–HAS_GENRE→Genre; builds a single NetworkX graph with weighted edges.
  2. **Node2Vec**: Runs Node2Vec (dimensions, walk length, etc. configurable); fits a model to get embeddings per node.
  3. **Export**: Maps node IDs to type (user/book/author/genre) and embedding vector; writes to Parquet (e.g. node_id, type, embedding columns).
  4. **Upload**: Uploads Parquet to S3 under `GRAPH_S3_PREFIX`.
- **entrypoint.sh**: Starts Ray head, runs `python graph_train.py`, stops Ray.
- **Recommendation service**: `graph_recommendation.py` uses Neo4j for the live graph and personalized PageRank; graph embeddings from S3 can be used for similarity or scoring (implementation may load this Parquet for book/user vectors).

## Environment

`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `S3_URI`, `GRAPH_S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RAY_ADDRESS`.
