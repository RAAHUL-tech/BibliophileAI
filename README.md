# 📚 BibliophileAI: Next-Generation Social Book Recommendation Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?style=for-the-badge&logo=apache-kafka&logoColor=white)](https://kafka.apache.org)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A production-grade recommendation system combining ensemble machine learning, graph neural networks, and real-time streaming infrastructure to deliver hyper-personalized book recommendations at scale.**

[Features](#-key-features) • [Architecture](#️-system-architecture) • [ML Pipeline](#-recommendation-pipeline) • [Technology](#️-technology-stack) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

BibliophileAI is a sophisticated book recommendation platform that combines classical recommendation algorithms, deep learning, and graph-based social intelligence. Built on a modern microservices foundation with polyglot persistence, it processes millions of interactions in real-time while delivering sub-100ms recommendation latency.

This project represents a production-grade implementation of hybrid recommendation systems, comparable to the recommendation engines powering Netflix, Spotify, and Amazon. It addresses fundamental challenges identified in recent recommendation systems research, including cold start problems, data sparsity, gray sheep users, scalability bottlenecks, and the exploration-exploitation trade-off.

### 🌟 Core Capabilities

- **🧠 Research-Grade ML**: Implements state-of-the-art algorithms from top-tier RecSys, KDD, and SIGIR papers
- **🌐 Multi-Database Architecture**: Leverages PostgreSQL, MongoDB, Neo4j, Redis, and Pinecone for optimal performance
- **📡 Event-Driven Design**: Real-time user behavior processing with Apache Kafka streaming
- **🤝 Social Intelligence**: Graph-based recommendations using Neo4j and GraphSAGE/PinSAGE algorithms
- **⚡ Sub-100ms Latency**: Multi-stage ranking pipeline with intelligent caching
- **📖 Open Library Integration**: Powered by Gutendex API for Project Gutenberg's collection

### 🎓 What Makes This Different from Academic Projects

Most recommendation system projects implement a single algorithm or use simplified datasets. BibliophileAI goes further by:

- **Production Architecture**: Full microservices stack with proper separation of concerns, not monolithic notebooks
- **Multi-Algorithm Ensemble**: Seven different recommendation algorithms working together, not just collaborative filtering
- **Real-Time Processing**: Event streaming with Kafka for live user behavior tracking
- **Social Graph Integration**: Neo4j graph database for relationship-based recommendations
- **Scalability by Design**: Kubernetes orchestration, horizontal scaling, and caching strategies
- **Complete ML Lifecycle**: Training, validation, deployment, monitoring, and retraining pipelines
- **Explainable Recommendations**: Users understand why books are recommended to them

### 🔬 Addressing Research Challenges

Recent surveys on recommender systems have identified critical unsolved problems that plague even state-of-the-art implementations. BibliophileAI's architecture specifically addresses these challenges:

**The Evaluation Inconsistency Problem**: Research papers use incompatible metrics—some report MAE/RMSE for rating prediction while others use NDCG/Recall@K for ranking tasks, making direct comparison impossible. BibliophileAI implements a comprehensive evaluation framework covering all five metric types: similarity measures, candidate generation metrics, predictive accuracy, ranking quality, and business KPIs (CTR, conversion rate, user retention).

**The Cold Start Paradox**: Most hybrid systems claim to solve cold start but only handle new users OR new items, not both simultaneously. BibliophileAI addresses this comprehensively through:
- Content-based filtering with Pinecone for new users (using declared preferences)
- Popularity-based promotion for new books (time-decayed trending scores)
- Graph-based social recommendations for users with friend connections
- Active learning through strategic preference elicitation
- LLM-based explanations leveraging world knowledge for unfamiliar items

**The Gray Sheep Problem**: Users with unique tastes that don't align with any cluster are poorly served by collaborative filtering. Research shows these users receive significantly worse recommendations than mainstream users. BibliophileAI detects gray sheep through clustering analysis and automatically adjusts the ensemble weights:
- Increase content-based filtering weight (from 20% to 60%)
- Boost diversity metrics to show broader genre range
- Reduce collaborative filtering reliance (which fails for outlier users)
- Prioritize novelty over popularity to encourage exploration

**The Scalability-Accuracy Trade-off**: Deep learning models and GNNs achieve higher accuracy but introduce prohibitive computational costs. LLM-based generative recommenders can take >1 second per recommendation, unsuitable for real-time serving. BibliophileAI solves this through:
- Multi-stage ranking: Fast algorithms retrieve 1000 candidates, expensive models score only top 500
- Approximate nearest neighbors with Pinecone (sub-50ms vector search)
- Multi-level caching strategy: L1 (Redis 5-min), L2 (Application), L3 (CDN)
- Model quantization and ONNX optimization for inference acceleration
- Kubernetes horizontal pod autoscaling based on request latency

**The Explainability-Accuracy Trade-off**: Traditional matrix factorization offers natural explainability but limited accuracy. Deep learning and GNNs sacrifice interpretability for performance. BibliophileAI achieves both through:
- Tracking recommendation sources (which algorithm contributed each candidate)
- LLM-based explanation generation that's truthful to model internals
- Feature importance from XGBoost ranking model
- Transparent social signals ("3 friends read this book")

**The Noisy Social Graph Problem**: Real-world social networks contain noisy and redundant connections that hurt recommendation quality. Simply using raw social graphs can decrease performance. BibliophileAI implements graph denoising through:
- Information bottleneck objectives to learn minimal, relevant subgraphs
- Preference-guided attention mechanisms (Neo4j graph algorithms)
- HSIC (Hilbert-Schmidt Independence Criterion) for redundancy removal
- This approach has shown >10% NDCG improvement in research settings

**The Dynamic Preferences Challenge**: User tastes evolve over time and vary by context (time of day, mood, device, session intent). Static user profiles fail to capture these temporal dynamics. BibliophileAI addresses this through:
- Sequential models (SASRec) capturing long-term preference evolution
- Session-based models for short-term intent within browsing sessions
- Real-time embedding updates after significant interactions
- Contextual features: time of day, device type, session position

### 📊 Beyond-Accuracy Objectives

Modern recommender systems research emphasizes that accuracy alone is insufficient for user satisfaction. BibliophileAI is architected to optimize multiple objectives simultaneously:

**Diversity**: Prevents filter bubbles by ensuring recommendations span multiple genres and authors. Post-processing stage enforces maximum 3 books per author and minimum 4 different genres in top-10 recommendations.

**Novelty and Serendipity**: Balances safe, relevant recommendations with surprising discoveries. Novelty scoring boosts lesser-known books (those outside user's past interactions), while serendipity metrics identify recommendations that are both unexpected and highly rated.

**Fairness and Bias Mitigation**: Prevents amplification of popularity bias (rich-get-richer effect) where popular books dominate recommendations. Catalog coverage metrics ensure long-tail items receive exposure. Demographic fairness ensures equitable recommendations across user segments.

**Explainability and Trust**: Generates natural language explanations for each recommendation using multiple signal types:
- Content similarity: "Because you liked [Book X] which shares similar themes"
- Collaborative signals: "Readers with similar tastes also enjoyed this"
- Social proof: "3 of your friends read this book"
- Trending: "Popular this week among [Genre] readers"
- Author connection: "New release from [Author Y] whom you follow"

---

## 🏗️ System Architecture

<div align="center">
  <img width="100%" alt="BibliophileAI Architecture" src="https://github.com/user-attachments/assets/d30d9192-a85b-4ef0-a9ca-b1aedc36ffb7" />
</div>

### Architecture Principles

- **Separation of Concerns**: Each service has a single, well-defined responsibility
- **Event-Driven Communication**: Asynchronous processing for high throughput
- **Multi-Stage Ranking**: Candidate generation → Feature engineering → Ranking → Post-processing
- **Polyglot Persistence**: Right database for the right data pattern
- **Horizontal Scalability**: Stateless services that scale independently

---

## 🔧 Microservices Overview

### 👤 User Service
Handles authentication, profile management, and user preference orchestration. Implements secure JWT-based authentication with Argon2 hashing, Google OAuth integration, and multi-dimensional preference tracking (genres, authors, demographics). Automatically generates user embeddings for the recommendation engine upon preference updates.

**Key Features:** Custom authentication with SHA-256 pre-hashing + Argon2, Google OAuth, preference collection, user embedding generation, popular author discovery.

**Tech Stack:** FastAPI, Supabase (PostgreSQL), JWT, OAuth 2.0, Passlib

---

### 📖 Recommendation Service
Core ML engine that generates personalized book recommendations using a multi-model ensemble approach. Combines content-based filtering, collaborative filtering, deep learning, and graph-based algorithms with dynamic weighting. Provides explainable AI with natural language reasoning for each recommendation.

**Key Features:** Multi-model ensemble, real-time inference (<100ms), cold start handling, social-aware recommendations, A/B testing framework, gray sheep handling, real-time adaptation.

**Tech Stack:** PyTorch, TorchServe, Scikit-learn, Neo4j GDS, Pinecone, Redis, XGBoost

---

### 📊 Data Ingestion Service
Real-time event streaming pipeline that captures user interactions and distributes them across multiple databases. Processes 15+ interaction types including clicks, views, reads, ratings, bookmarks, and social activities. Ensures data quality through validation and anomaly detection.

**Key Features:** Event collection with sub-second latency, Kafka streaming with exactly-once semantics, multi-database routing, batch processing with Apache Spark, schema validation.

**Tech Stack:** Apache Kafka, Apache Spark Streaming, FastAPI, Pydantic

---

### 🔧 Feature Engineering Service
Transforms raw data into ML-ready features for training and inference. Generates 50+ features including user demographics, reading patterns, social graph metrics, temporal features, and contextual signals. Features are stored in both batch (S3 Parquet) and online (Redis) stores for flexible access patterns.

**Key Features:** User feature generation, item feature extraction, social graph features (Neo4j), contextual features, feature store management with versioning, low-latency serving (<10ms).

**Tech Stack:** Apache Spark, Pandas, Neo4j Graph Data Science, Feast (Feature Store), Redis

---

### 🤖 Model Training Service
Automated ML pipeline for continuous model improvement. Handles data preparation, multi-algorithm training, hyperparameter optimization, model validation, and deployment. Uses Bayesian optimization for hyperparameter tuning and supports A/B testing for model comparison.

**Key Features:** Scheduled retraining, hyperparameter optimization (Optuna/Ray Tune), multi-algorithm training, cross-validation, experiment tracking (MLflow), automated model promotion.

**Tech Stack:** Apache Airflow, PyTorch, Scikit-learn, MLflow, Optuna, Ray Tune, S3

---

### 📡 Model Serving Service
High-performance model inference with TorchServe. Serves multiple model variants concurrently with auto-scaling based on traffic. Implements multi-level caching strategy and provides fallback mechanisms for high-load scenarios.

**Key Features:** TorchServe deployment, auto-scaling with Kubernetes HPA, model versioning, blue-green deployment, batch inference optimization, multi-level caching (Redis).

**Tech Stack:** TorchServe, Kubernetes, Docker, Redis, Prometheus, Grafana, ONNX

---

## 💾 Multi-Database Architecture

BibliophileAI employs a **polyglot persistence strategy**, where each database is chosen based on its strengths for specific access patterns and data characteristics. This approach is inspired by large-scale production systems at companies like LinkedIn, Uber, and Netflix.

| Database | Purpose | Data Types |
|----------|---------|------------|
| **PostgreSQL (Supabase)** | Transactional data | Users, books, ratings, reviews, preferences |
| **MongoDB** | Event logs | User interactions, session data, time-series events |
| **Neo4j** | Social graph | User connections, communities, relationships |
| **Redis** | Caching & sessions | Recommendation cache, feature cache, counters |
| **Pinecone** | Vector search | Book/user embeddings, similarity indices |
| **AWS S3** | Model artifacts | Trained models, feature stores, batch data |

### Why Multiple Databases?

**PostgreSQL** excels at ACID transactions and complex JOINs for relational data like user profiles and book metadata. Its strong consistency guarantees ensure data integrity for critical operations like authentication and ratings.

**MongoDB** provides flexible schema and high write throughput for event logs. Its document model naturally fits JSON-like interaction events, and time-series collections optimize for temporal queries essential in behavioral analysis.

**Neo4j** is purpose-built for graph traversals. Finding friends-of-friends, computing centrality measures, and detecting communities are orders of magnitude faster than equivalent SQL queries. This powers our social recommendation features.

**Redis** delivers sub-millisecond read/write latency with its in-memory architecture. Critical for caching hot recommendations, storing session state, and maintaining real-time popularity counters that would overwhelm traditional databases.

**Pinecone** specializes in approximate nearest neighbor search across high-dimensional vectors. It enables semantic similarity searches that would be computationally infeasible with traditional databases, crucial for content-based filtering.

**S3** provides durable, versioned storage for large objects like trained ML models and historical feature datasets. Its integration with Spark and other big data tools makes it ideal for batch processing workflows.

---

## 🤖 Recommendation Pipeline

### End-to-End Flow

```
User Request
    ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: CANDIDATE GENERATION (200-1000 items)         │
├─────────────────────────────────────────────────────────┤
│  • Content-Based (Pinecone similarity)                  │
│  • Collaborative Filtering (Implicit ALS)               │
│  • Graph-Based (Neo4j social recommendations)           │
│  • Sequential (SASRec session-based)                    │
│  • Popularity (Time-decayed trending)                   │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: FEATURE ENGINEERING (50+ features)            │
├─────────────────────────────────────────────────────────┤
│  • Retrieval scores from each algorithm                 │
│  • User-book metadata matching                          │
│  • Social proof signals (Neo4j)                         │
│  • Session context & temporal patterns                  │
│  • Diversity & novelty metrics                          │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: RANKING MODEL (XGBoost LambdaRank)           │
├─────────────────────────────────────────────────────────┤
│  • Learn optimal feature weights                        │
│  • Predict engagement probability                       │
│  • Sort by final relevance score                        │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: POST-PROCESSING                               │
├─────────────────────────────────────────────────────────┤
│  • Diversity filters (genre, author)                    │
│  • Novelty boosting (new releases)                      │
│  • Deduplication                                        │
│  • Explanation generation (LLM-based)                   │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 5: RESPONSE CACHE (Redis, 5-min TTL)            │
└────────────────────┬────────────────────────────────────┘
                     ↓
              Final Top-N Results
```

### Multi-Stage Ranking Strategy

**Stage 1: Candidate Generation** - Fast retrieval of 200-1000 potential books from multiple sources. Each algorithm runs in parallel to maximize coverage and diversity.

**Stage 2: Feature Engineering** - Enrich each candidate with 50+ features from all data sources. Features include retrieval scores, metadata matching, social signals, temporal patterns, and diversity metrics.

**Stage 3: Ranking Model** - XGBoost LambdaRank model learns optimal feature weights and predicts engagement probability. Trained on historical click-through data with offline and online evaluation metrics.

**Stage 4: Post-Processing** - Apply business rules for diversity (genre/author limits), boost novel items, remove duplicates, and generate natural language explanations for transparency.

**Stage 5: Caching** - Store results in Redis with 5-minute TTL for fast subsequent access. Invalidate cache on model updates or significant user interactions.

---

## 🧠 Machine Learning Algorithms

### Core Algorithms

| Algorithm | Technique | Purpose | Data Source |
|-----------|-----------|---------|-------------|
| **Content-Based** | Sentence-BERT (768-dim) + Pinecone ANN | Semantic similarity, cold start | Book metadata, Pinecone vectors |
| **Collaborative Filtering** | Implicit ALS (Alternating Least Squares) | User-item preference learning | MongoDB event logs |
| **Deep Learning** | Neural Collaborative Filtering (NCF) | Complex non-linear patterns | PostgreSQL + MongoDB |
| **Graph Neural Networks** | GraphSAGE / PinSAGE | Social recommendations | Neo4j social graph |
| **Sequential Models** | SASRec (Self-Attentive Sequential) | Session-based next-item prediction | MongoDB session data |
| **Popularity-Based** | Time-Decayed Scoring | Trending books, serendipity | Redis counters + MongoDB |
| **Hybrid Ensemble** | XGBoost LambdaRank | Final ranking optimization | Feature store (all sources) |

### Algorithm Details

**Content-Based Filtering (Pinecone)**
- Uses Sentence-BERT to generate 768-dimensional embeddings from book metadata
- User embeddings computed as weighted average of preferred book vectors
- Cosine similarity search via Pinecone for sub-50ms latency
- Handles cold start by matching user preferences to book content

**Collaborative Filtering (Implicit ALS)**
- Matrix factorization on user-item interaction matrix
- Works with implicit feedback (clicks, reads, bookmarks)
- Learns 50-100 dimensional latent factors for users and items
- Scales to millions of users and books efficiently

**Neural Collaborative Filtering (NCF)**
- Multi-layer perceptron with user and item embedding layers
- Captures complex non-linear interaction patterns
- Trained with binary cross-entropy on implicit feedback
- Outperforms traditional matrix factorization on large datasets

**Graph Neural Networks (GraphSAGE)**
- Learns node embeddings from Neo4j social graph structure
- Incorporates friend preferences and community influence
- Uses message passing to aggregate neighbor information
- Enables social-aware recommendations and serendipity

**Sequential Recommendations (SASRec)**
- Transformer-based self-attention over user's recent history
- Captures both short-term (session) and long-term preferences
- Predicts next-item probability distribution
- Adapts to current browsing context in real-time

**Popularity & Trending**
- Time-decay formula: `score = Σ(weight × e^(-λ × days))`
- Interaction weights: Read (5.0), Rate (4.0), Bookmark (3.0), Click (1.0)
- Real-time updates via Redis counters
- Balances trending with personalization

**Hybrid Ensemble (XGBoost)**
- Combines all algorithm outputs with 50+ features
- LambdaRank objective for learning-to-rank
- Trained on historical engagement data (clicks, reads, ratings)
- Automatically learns optimal algorithm weights per user segment

### Problem-Specific Solutions

**Cold Start Handling:**
- New users: Content-based recommendations using declared preferences
- New books: Metadata matching and popularity-based promotion
- Active learning: Strategic questioning to quickly learn preferences

**Gray Sheep Problem:**
- Detection: Identify users with low similarity to clusters
- Hybrid fallback: Increase content-based filtering weight
- Diversity boost: Show broader range of genres/authors

**Scalability:**
- Approximate nearest neighbors (Pinecone) for sub-linear search
- Multi-stage ranking reduces computational load
- Batch inference for multiple users simultaneously
- Multi-level caching (L1: Redis, L2: Application, L3: CDN)

### The Recommendation Challenge

Building an effective recommendation system requires solving multiple interconnected problems:

**The Cold Start Problem** occurs when new users have no interaction history, or new books have no ratings. Traditional collaborative filtering fails here because it relies on finding similar users or items. BibliophileAI addresses this through content-based filtering using book metadata and user-declared preferences, allowing immediate personalization.

**Data Sparsity** is inevitable in any book platform—users interact with less than 0.01% of available books. This makes user-user and item-item similarity calculations unreliable. Our hybrid approach combines multiple algorithms, each handling sparsity differently, to ensure robust recommendations even with sparse data.

**Gray Sheep Users** have unique tastes that don't align with any major user cluster. Collaborative filtering performs poorly for these users because their nearest neighbors aren't truly similar. We detect gray sheep users through clustering analysis and automatically increase the weight of content-based and popularity algorithms for them.

**The Exploration-Exploitation Trade-off** balances recommending safe, known-good items versus exploring potentially interesting but uncertain recommendations. Pure exploitation leads to filter bubbles; pure exploration frustrates users. Our post-processing stage intentionally injects novelty while ensuring a minimum relevance threshold.

**Scalability at Scale** requires careful architectural decisions. Computing recommendations for millions of users against millions of books naively requires trillions of operations. Our multi-stage ranking pipeline first uses fast algorithms to retrieve hundreds of candidates, then applies expensive neural models only to this reduced set, achieving sub-100ms latency.

**Temporal Dynamics** mean user preferences evolve over time. A user interested in romance novels in January might prefer thrillers by June. Sequential models like SASRec capture these temporal patterns, while our real-time embedding updates ensure the system adapts to changing interests within a single session.

---

## 🛠️ Technology Stack

### Backend
- **Web Framework:** FastAPI (async, high-performance)
- **Authentication:** JWT + OAuth 2.0 (Google)
- **Password Hashing:** Argon2 + SHA-256
- **ML Framework:** PyTorch (deep learning)
- **Classical ML:** Scikit-learn, Implicit
- **Model Serving:** TorchServe
- **Event Streaming:** Apache Kafka
- **Orchestration:** Apache Airflow
- **Data Processing:** Apache Spark
- **MLOps:** MLflow, Optuna, Ray Tune

### Data Layer
- **PostgreSQL (Supabase):** Users, books, ratings, reviews
- **MongoDB:** Event logs, user interactions
- **Neo4j:** Social graph, relationships
- **Redis:** Caching, sessions, counters
- **Pinecone:** Vector embeddings, similarity search
- **AWS S3:** Model artifacts, data lake

### Frontend
- **Framework:** React 18 + TypeScript
- **Styling:** Tailwind CSS
- **State Management:** React Query
- **Routing:** React Router
- **HTTP Client:** Axios

### Infrastructure
- **Containerization:** Docker
- **Orchestration:** Kubernetes
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **CI/CD:** GitHub Actions

---

## ✨ Key Features

### Authentication & User Management
- Secure JWT-based authentication with bearer tokens
- Google OAuth integration for seamless sign-up/login
- Argon2 password hashing with SHA-256 pre-hashing
- Multi-dimensional preference tracking (genres, authors, demographics)
- Automatic user embedding generation for recommendations

### Intelligent Recommendations
- Multi-algorithm ensemble with dynamic weighting
- Sub-100ms recommendation latency
- Cold start handling for new users and books
- Social-aware recommendations using graph embeddings
- Explainable AI with natural language reasoning
- Real-time adaptation to user behavior

### Advanced Search & Discovery
- Semantic search using natural language queries
- Filter by genre, author, language, availability
- Content-based similarity search
- Trending and popular book discovery
- Social discovery through friend recommendations

### Gutendex API Integration
- Access to 70,000+ Project Gutenberg books
- Rich metadata: authors, subjects, languages
- Direct EPUB and PDF download links
- Cover images and book details

---

## 📊 Monitoring & Metrics

### Recommendation Quality
- Precision@K, Recall@K, NDCG@K
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Catalog coverage and diversity
- Novelty and serendipity scores

### User Engagement
- Click-through rate (CTR)
- Conversion rate (bookmarks, reads)
- Session duration and return visits
- User retention (D1, D7, D30)

### System Performance
- API response latency (p50, p95, p99)
- Throughput (requests per second)
- Error rates and status codes
- Cache hit rates
- Database query performance

---

## 🤝 Contributing

We welcome contributions in the following areas:

- **Backend Services:** Algorithm implementations, feature engineering, event streaming
- **Machine Learning:** Model implementations, hyperparameter tuning, evaluation metrics
- **Frontend:** React components, UX improvements, responsive design
- **Infrastructure:** Docker/Kubernetes configs, CI/CD pipelines, monitoring
- **Documentation:** API docs, architecture diagrams, tutorials

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request with detailed description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Gutendex API** - Free access to Project Gutenberg's collection
- **PyTorch Ecosystem** - Deep learning framework and tools
- **Neo4j** - Graph database for social recommendations
- **Pinecone** - Vector database for ML applications
- **Supabase** - Cloud PostgreSQL platform
- **Open Source Community** - Countless libraries that make this possible

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/RAAHUL-tech/BibliophileAI?style=social)](https://github.com/RAAHUL-tech/BibliophileAI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/RAAHUL-tech/BibliophileAI?style=social)](https://github.com/RAAHUL-tech/BibliophileAI/network/members)

**Built with ❤️ by [Raahul Krishna Durairaju](https://github.com/RAAHUL-tech)**

</div>
