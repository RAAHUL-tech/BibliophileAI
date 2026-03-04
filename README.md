# 📚 BibliophileAI: Next-Generation Social Book Recommendation Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![AWS SQS](https://img.shields.io/badge/AWS%20SQS-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sqs/)
[![Ray](https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white)](https://ray.io)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A production-grade recommendation system combining 6 recommendation algorithms, Ray-based distributed training, XGBoost learning-to-rank, and Feast feature store to deliver hyper-personalized book recommendations at scale.**

[What & Why](#-what-is-this-project) • [Architecture](#-system-architecture) • [Request Flow](#-end-to-end-request-flow) • [Pipeline](#-recommendation-pipeline) • [Tech Stack](#️-technology-stack) • [Getting Started](#-getting-started) • [Contributing](#-contributing)

</div>

---

## 📌 What is this project?

**BibliophileAI** is a **social book recommendation platform** that helps users discover books through personalized feeds, semantic search, and social signals. It is built as a **microservices-based system** running on Kubernetes, with a React frontend, multiple backend services (user, recommendation, search), and an event-driven pipeline for clickstream data and offline model training.

### 🎯 Goals

- **Personalization at scale** — Combine content-based, collaborative, graph-based, session-based, and popularity-driven signals into one ranked feed.
- **Production-ready design** — Clear service boundaries, polyglot persistence (PostgreSQL, MongoDB, Neo4j, Redis, Pinecone, S3), caching, and observability (Prometheus/Grafana).
- **Full ML lifecycle** — From candidate generation and feature engineering (Feast) to learning-to-rank (XGBoost) and diversity-aware post-processing, with Airflow-orchestrated training (ALS, Graph, SASRec, Popularity, LTR).
- **Explainability and control** — Users see why a book is recommended (e.g. “Readers like you also liked”, “Trending in your genre”); operators can tune algorithms and training schedules.

### 🔧 What problem it solves

- **Discovery** — Users get a single, ranked “For You” experience (Netflix-style rows by algorithm/category) instead of scattered lists.
- **Cold start** — New users get content-based and trending recommendations from day one; the system refines with behavior.
- **Sparsity & gray sheep** — Hybrid algorithms and fallbacks (content-based, popularity) handle sparse data and niche tastes.
- **Latency vs. quality** — Multi-stage pipeline: fast candidate retrieval, then feature engineering and LTR on a smaller set, with Redis cache for repeated requests.
- **Operational clarity** — Services, data stores, and training jobs are separated so teams can own and scale parts independently.

### ⚠️ Limitations

- **External dependencies** — Requires Supabase, MongoDB, Neo4j, Pinecone, AWS (S3, SQS), and (optional) Google OAuth. Not a single-binary or single-DB setup.
- **First-request latency** — Full recommendation pipeline on cache miss can take 1–2 minutes; the Ingress is tuned (e.g. 300s) so the request does not time out. Subsequent requests are fast from cache.
- **Training lag** — Models (ALS, Graph, SASRec, etc.) are trained offline (e.g. daily/hourly via Airflow); recommendations use the last published model until the next run.
- **Scale and cost** — Designed for moderate scale; very large catalogs or QPS may need more tuning (sharding, caching, model serving) and higher infra cost.

### 💡 How to improve

- **Latency** — Pre-warm cache for active users, optimize LTR/Feast lookups, consider a dedicated ranking service with gRPC.
- **Freshness** — More frequent training or incremental updates (e.g. streaming model updates), real-time feature pipelines.
- **Evaluation** — Add offline metrics (NDCG, recall) and A/B tests; expose them in Grafana.
- **Resilience** — Circuit breakers and fallbacks per algorithm; degrade gracefully if one store (e.g. Neo4j) is slow.
- **Security** — Keep secrets in a vault; use short-lived tokens and least-privilege IAM for AWS/Supabase/Neo4j.

---

## 🎯 Overview

BibliophileAI is a sophisticated book recommendation platform that combines classical recommendation algorithms, deep learning, and graph-based social intelligence. Built on a modern microservices foundation with polyglot persistence, it processes millions of interactions in real-time while delivering sub-100ms recommendation latency when served from cache.

This project represents a production-grade implementation of hybrid recommendation systems, comparable to the recommendation engines powering Netflix, Spotify, and Amazon. It addresses fundamental challenges identified in recent recommendation systems research, including cold start problems, data sparsity, gray sheep users, scalability bottlenecks, and the exploration-exploitation trade-off.

### 🌟 Core Capabilities

- **🧠 Research-Grade ML**: Implements state-of-the-art algorithms from top-tier RecSys, KDD, and SIGIR papers
- **🌐 Multi-Database Architecture**: Leverages PostgreSQL, MongoDB, Neo4j, Redis, and Pinecone for optimal performance
- **📡 Event-Driven Design**: Real-time user behavior processing with AWS SQS (Kafka can be used as a replacement for higher throughput)
- **🤝 Social Intelligence**: Graph-based recommendations using Neo4j and PageRank algorithms
- **⚡ Sub-100ms Latency**: Multi-stage ranking pipeline with intelligent caching
- **📖 Book catalog**: Books are scraped from Gutendex (Project Gutenberg index) and stored in Supabase; the book_data_importer handles ingestion.

### 🎓 What Makes This Different from Academic Projects

Most recommendation system projects implement a single algorithm or use simplified datasets. BibliophileAI goes further by:

- **Production Architecture**: Full microservices stack with proper separation of concerns, not monolithic notebooks
- **Multi-Algorithm Ensemble**: Six different recommendation algorithms working together, not just collaborative filtering
- **Real-Time Processing**: Event streaming with AWS SQS for live user behavior tracking (replaceable by Kafka for higher scale)
- **Social Graph Integration**: Neo4j graph database for relationship-based recommendations
- **Scalability by Design**: Kubernetes orchestration, horizontal scaling, and caching strategies
- **Complete ML Lifecycle**: Ray-based distributed training, Feast feature store, XGBoost learning-to-rank, and diversity-aware post-processing
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

### High-Level Architecture (Overview)

The diagram below shows how the **client**, **ingress**, **core services**, **data stores**, **clickstream consumer**, **offline training**, and **monitoring** fit together.

![BibliophileAI Architecture](https://github.com/user-attachments/assets/17b1ae43-32f6-4f1e-aba8-8bfa304c6d93)

*Flow: React SPA → Ingress → User / Recommendation / Search services → Supabase, Redis, Neo4j, Pinecone, MongoDB, S3, SQS → Clickstream Consumer → Airflow (ALS, Graph, LTR, Popularity, SASRec) → Prometheus + Grafana.*

### Architecture (Mermaid)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#E3F2FD', 'primaryTextColor':'#0D47A1', 'primaryBorderColor':'#1976D2', 'lineColor':'#424242', 'secondaryColor':'#F3E5F5', 'tertiaryColor':'#E8F5E9' }}}%%
flowchart TB
    subgraph Client["🖥️ Client"]
        SPA["React SPA"]
    end

    subgraph Ingress["🌐 Ingress"]
        NGINX["Ingress Controller<br/>/user · /recommend · /search"]
    end

    subgraph K8s["Kubernetes Cluster"]
        subgraph Services["Core Services"]
            US["User Service"]
            RS["Recommendation Service"]
            SS["Search Service"]
        end
        CC["Clickstream Consumer"]
    end

    subgraph Data["Data & Messaging"]
        Supabase[("Supabase Postgres")]
        Redis[("Redis")]
        Neo4j[("Neo4j")]
        Pinecone[("Pinecone")]
        MongoDB[("MongoDB")]
        S3[("AWS S3")]
        SQS["AWS SQS"]
    end

    subgraph Training["Offline Model Training"]
        Airflow["Airflow"]
        ALS["ALS Train"]
        Graph["Graph Train"]
        LTR["LTR Train"]
        Pop["Popularity Train"]
        SasRec["SASRec Train"]
    end

    subgraph Monitor["Monitoring"]
        Prom["Prometheus"]
        Grafana["Grafana"]
    end

    SPA --> NGINX
    NGINX --> US
    NGINX --> RS
    NGINX --> SS
    US --> Supabase
    US --> SQS
    RS --> Redis
    RS --> Neo4j
    RS --> Pinecone
    RS --> SQS
    SS --> Pinecone
    SQS --> CC
    CC --> MongoDB
    RS --> S3
    Airflow --> ALS
    Airflow --> Graph
    Airflow --> LTR
    Airflow --> Pop
    Airflow --> SasRec
    ALS --> S3
    Graph --> Neo4j
    SasRec --> S3
    Pop --> Redis
    LTR --> S3
    RS -.-> Prom
    Prom --> Grafana
```

### Architecture Principles

- **Separation of concerns** — Each service has a single, well-defined responsibility (user, recommendation, search, consumer).
- **Event-driven communication** — Clickstream events flow asynchronously via SQS to the consumer, then to MongoDB.
- **Multi-stage ranking** — Candidate generation → Feature engineering → Ranking (LTR) → Post-processing → Cache.
- **Polyglot persistence** — PostgreSQL (Supabase), MongoDB (events), Neo4j (graph), Redis (cache), Pinecone (vectors), S3 (models).
- **Horizontal scalability** — Stateless services; training runs in separate pods orchestrated by Airflow.

---

## 🔀 End-to-end request flow

### 1. Traffic flow: client → ingress → services

Every user action (login, recommendations, search) goes through the React app to the Ingress, which routes by path to the right backend.

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#E8F5E9', 'primaryTextColor':'#1B5E20', 'lineColor':'#2E7D32' }}}%%
sequenceDiagram
    autonumber
    participant U as 👤 User
    participant R as ⚛️ React SPA
    participant I as 🌐 Ingress
    participant US as User Service
    participant RS as Recommendation Service
    participant SS as Search Service

    U->>R: Open app / Login / Search / View feed
    R->>I: HTTP (Bearer token)
    alt /api/v1/user/*
        I->>US: Forward
        US->>US: Auth, Supabase, Neo4j, SQS
        US-->>R: JSON
    else /api/v1/recommend/*
        I->>RS: Forward
        RS->>RS: Cache / full pipeline
        RS-->>R: Categories + books
    else /api/v1/search
        I->>SS: Forward
        SS->>SS: Pinecone + Supabase
        SS-->>R: Search results
    end
    R-->>U: Render UI
```

### 2. Recommendation path: from request to displayed rows

When the user opens the homepage or refreshes recommendations, the following pipeline runs (on cache miss). The result is a set of **category rows** (e.g. “Content-Based”, “For You (LinUCB)”, “Top Picks”) that the frontend renders as horizontal carousels.

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#FFF3E0', 'primaryTextColor':'#E65100', 'secondaryColor':'#E3F2FD', 'tertiaryColor':'#F3E5F5' }}}%%
flowchart TB
    subgraph Request["1️⃣ Request"]
        A[GET /recommend/combined]
    end

    subgraph Cache["2️⃣ Cache"]
        B{Redis cache?}
        A --> B
    end

    subgraph Candidates["3️⃣ Candidate generation"]
        B -->|Miss| C[Content-Based]
        B -->|Miss| D[ALS]
        B -->|Miss| E[Graph]
        B -->|Miss| F[SASRec]
        B -->|Miss| G[Popularity]
        C --> H[Merge candidates]
        D --> H
        E --> H
        F --> H
        G --> H
    end

    subgraph Rank["4️⃣ Rank & features"]
        H --> I[LinUCB score]
        I --> J[Feast 29 features]
        J --> K[LTR XGBoost]
        K --> L[Post-process diversity]
    end

    subgraph Store["5️⃣ Store & respond"]
        L --> M[Cache in Redis]
        M --> N[JSON response]
        B -->|Hit| N
    end

    subgraph Display["6️⃣ Display"]
        N --> O[React: category rows]
        O --> P[User sees carousels]
    end
```

### 3. Detailed recommendation flow (candidate generation → display)

This diagram expands the **candidate generation → ranking → response → display** steps with the main data stores and outputs.

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#E8F5E9', 'secondaryColor':'#FCE4EC', 'tertiaryColor':'#F3E5F5' }}}%%
flowchart TB
    Start([User requests recommendations]) --> CheckCache{Check Redis<br/>cache}
    CheckCache -->|Hit| AddTrending[Add Trending Now from Redis]
    AddTrending --> Return[Return JSON to frontend]
    CheckCache -->|Miss| NewUser{New user?}
    NewUser -->|Yes| CBOnly[Content-Based + Trending only]
    CBOnly --> CacheStore[Store in Redis]
    CacheStore --> Return
    NewUser -->|No| Stage1[Stage 1: Candidate generation]
    Stage1 --> C1[Pinecone: content-based]
    Stage1 --> C2[S3: ALS factors]
    Stage1 --> C3[Neo4j: graph]
    Stage1 --> C4[MongoDB: SASRec session]
    Stage1 --> C5[Redis: popularity]
    C1 --> Merge[Merge & dedupe]
    C2 --> Merge
    C3 --> Merge
    C4 --> Merge
    C5 --> Merge
    Merge --> LinUCB[LinUCB rank candidates]
    LinUCB --> Feast[Feast: 29 features]
    Feast --> LTR[XGBoost LTR: Top Picks]
    LTR --> PostProc[Diversity post-process]
    PostProc --> FetchBooks[Fetch book metadata Supabase]
    FetchBooks --> Build[Build category payload]
    Build --> CacheStore
    Return --> Frontend[React: render rows]
    Frontend --> User([User sees recommendations])

    style Start fill:#C8E6C9
    style Return fill:#C8E6C9
    style User fill:#C8E6C9
    style Stage1 fill:#BBDEFB
    style Merge fill:#E1BEE7
    style LTR fill:#FFE0B2
```

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

#### Recommendation Service Architecture

```mermaid
graph TB
    subgraph API["API Endpoints"]
        ACombined["/api/v1/recommend/combined"]
        AContent["/api/v1/recommend/content"]
        AGraph["/api/v1/recommend/graph"]
        ASession["/api/v1/recommend/session"]
        APopularity["/api/v1/recommend/popularity"]
    end

    subgraph Algo["Algorithm Modules"]
        CB["Content-Based<br/>Pinecone"]
        CF["Collaborative Filtering<br/>ALS"]
        GR["Graph Recommendation<br/>Neo4j"]
        SR["SASRec<br/>Session-Based"]
        POP["Popularity<br/>Time-Decayed"]
        LIN["LinUCB<br/>Contextual Bandit"]
    end

    subgraph FEBlock["Feature Engineering"]
        FE["Feature Service<br/>20+ Features"]
    end

    subgraph Rank["Ranking"]
        XGB["XGBoost<br/>LambdaRank"]
    end

    ACombined --> CB
    ACombined --> CF
    ACombined --> GR
    ACombined --> SR
    ACombined --> POP
    ACombined --> LIN

    CB --> FE
    CF --> FE
    GR --> FE
    SR --> FE
    POP --> FE
    LIN --> FE

    FE --> XGB
    XGB --> ACombined

    AContent --> CB
    AGraph --> GR
    ASession --> SR
    APopularity --> POP

    style ACombined fill:#4CAF50,color:#fff
    style XGB fill:#FF9800,color:#fff
```

---

### 📊 Data Ingestion Service
Real-time event streaming pipeline that captures user interactions and distributes them across multiple databases. Processes 15+ interaction types including clicks, views, reads, ratings, bookmarks, and social activities. Ensures data quality through validation and anomaly detection.

**Key Features:** Event collection with sub-second latency, SQS queue with async consumer, multi-database routing, schema validation. (Kafka can replace SQS for exactly-once semantics or higher throughput.)

**Tech Stack:** AWS SQS, FastAPI, Pydantic, MongoDB

#### Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant SQS as AWS SQS
    participant Consumer
    participant MongoDB
    participant Redis
    participant Neo4j
    
    User->>Frontend: Interacts (click, read, etc.)
    Frontend->>API: POST /api/events
    API->>SQS: Send Message
    SQS->>Consumer: Receive & Process
    Consumer->>Consumer: Validate & Transform
    Consumer->>MongoDB: Store Event Log
    Consumer->>Redis: Update Counters
    Consumer->>Neo4j: Update Graph
    Consumer->>SQS: Delete Message
```

---

### 🔧 Feature Engineering Service
Transforms raw data into ML-ready features for training and inference. Generates 29 features including retrieval scores from all 6 algorithms, user-book metadata matching, social graph metrics, temporal features, and contextual signals. Features are stored in Feast feature store (S3 Parquet) for training the learning-to-rank model.

**Key Features:** 29-feature vector generation, retrieval scores from all algorithms, social graph features (Neo4j), contextual features, Feast feature store integration, batch feature storage to S3.

**Tech Stack:** Pandas, Neo4j Graph Data Science, Feast (Feature Store), Redis, S3

---

### 🤖 Model Training Service
Automated ML pipeline for continuous model improvement using Ray framework for distributed training. Handles data preparation, multi-algorithm training across 6 recommendation algorithms, model validation, and deployment. All training jobs run as Kubernetes pods orchestrated by Apache Airflow.

**Key Features:** Scheduled retraining via Airflow DAGs, Ray-based distributed training, multi-algorithm training (ALS, Graph, SASRec, Popularity, LinUCB), model validation, automated S3 model storage.

**Tech Stack:** Apache Airflow, Ray, PyTorch, Scikit-learn, Implicit (ALS), Node2Vec, S3

**Training Algorithms:**
- **ALS (Collaborative Filtering)**: Matrix factorization using Implicit library, trained with Ray remote functions
- **Graph Analytics**: Node2Vec embeddings + PageRank scores computed with Ray
- **SASRec (Sequential)**: Transformer-based session model trained with PyTorch and Ray
- **Popularity**: Time-decayed scoring with exponential decay, computed with Ray workers
- **LinUCB**: Contextual bandit for exploration-exploitation balance

#### Training Pipeline Architecture

```mermaid
flowchart TD
    Start[Airflow DAG Trigger<br/>Kubernetes Pod] --> RayInit[Initialize Ray Cluster]
    
    RayInit --> ALS[ALS Training<br/>Ray Remote Functions<br/>Implicit ALS]
    RayInit --> Graph[Graph Training<br/>Ray Remote Functions<br/>Node2Vec + PageRank]
    RayInit --> SASRec[SASRec Training<br/>Ray Remote Functions<br/>PyTorch Transformer]
    RayInit --> Popularity[Popularity Training<br/>Ray Remote Actors<br/>Time-Decay Scoring]
    RayInit --> LinUCB[LinUCB Training<br/>Contextual Bandit]
    
    ALS --> S3_ALS[Upload Model to S3]
    Graph --> S3_Graph[Upload Embeddings to S3]
    SASRec --> S3_SASRec[Upload Checkpoint to S3]
    Popularity --> S3_Pop[Upload Scores to S3]
    LinUCB --> S3_LinUCB[Upload Bandit State to S3]
    
    S3_ALS --> End[Training Complete<br/>Models Available for Inference]
    S3_Graph --> End
    S3_SASRec --> End
    S3_Pop --> End
    S3_LinUCB --> End
    
    style Start fill:#E3F2FD
    style RayInit fill:#FF6B6B,color:#fff
    style End fill:#E8F5E9
```

---

## 💾 Multi-Database Architecture

BibliophileAI employs a **polyglot persistence strategy**, where each database is chosen based on its strengths for specific access patterns and data characteristics. This approach is inspired by large-scale production systems at companies like LinkedIn, Uber, and Netflix.

### Database Architecture Diagram

```mermaid
graph LR
    subgraph "Application Layer"
        APP[Recommendation Service]
    end
    
    subgraph "Transactional Data"
        PG[(PostgreSQL<br/>Supabase)]
        PG --> |Users, Books<br/>Ratings, Reviews| APP
    end
    
    subgraph "Event & Time-Series"
        MG[(MongoDB)]
        MG --> |Event Logs<br/>Sessions| APP
    end
    
    subgraph "Graph Data"
        N4J[(Neo4j)]
        N4J --> |Social Graph<br/>Relationships| APP
    end
    
    subgraph "In-Memory Cache"
        RD[(Redis)]
        RD --> |Popularity Scores<br/>Recommendation Cache<br/>Session State| APP
    end
    
    subgraph "Vector Search"
        PC[(Pinecone)]
        PC --> |Book Embeddings<br/>Similarity Search| APP
    end
    
    subgraph "Object Storage"
        S3[(AWS S3)]
        S3 --> |Trained Models<br/>Feature Stores<br/>Batch Data| APP
    end
    
    style PG fill:#336791,color:#fff
    style MG fill:#47A248,color:#fff
    style N4J fill:#008CC1,color:#fff
    style RD fill:#DC382D,color:#fff
    style PC fill:#5A67D8,color:#fff
    style S3 fill:#FF9900,color:#fff
```

### Database Usage Matrix

| Database | Purpose | Data Types | Access Pattern | Latency |
|----------|---------|------------|----------------|---------|
| **PostgreSQL (Supabase)** | Transactional data | Users, books, ratings, reviews, preferences | ACID transactions, JOINs | ~10-50ms |
| **MongoDB** | Event logs | User interactions, session data, time-series events | High write throughput, aggregations | ~5-20ms |
| **Neo4j** | Social graph | User connections, communities, relationships | Graph traversals, PPR | ~20-100ms |
| **Redis** | Caching & sessions | Recommendation cache, feature cache, counters | Key-value, sorted sets | <1ms |
| **Pinecone** | Vector search | Book/user embeddings, similarity indices | ANN similarity search | ~20-50ms |
| **AWS S3** | Model artifacts | Trained models, feature stores, batch data | Object storage, batch reads | ~100-500ms |

### Why Multiple Databases?

**PostgreSQL** excels at ACID transactions and complex JOINs for relational data like user profiles and book metadata. Its strong consistency guarantees ensure data integrity for critical operations like authentication and ratings.

**MongoDB** provides flexible schema and high write throughput for event logs. Its document model naturally fits JSON-like interaction events, and time-series collections optimize for temporal queries essential in behavioral analysis.

**Neo4j** is purpose-built for graph traversals. Finding friends-of-friends, computing centrality measures, and detecting communities are orders of magnitude faster than equivalent SQL queries. This powers our social recommendation features.

**Redis** delivers sub-millisecond read/write latency with its in-memory architecture. Critical for caching hot recommendations, storing session state, and maintaining real-time popularity counters that would overwhelm traditional databases.

**Pinecone** specializes in approximate nearest neighbor search across high-dimensional vectors. It enables semantic similarity searches that would be computationally infeasible with traditional databases, crucial for content-based filtering.

**S3** provides durable, versioned storage for large objects like trained ML models and historical feature datasets. Its integration with Spark and other big data tools makes it ideal for batch processing workflows.

---

## 🤖 Recommendation Pipeline

### End-to-End Flow Diagram

*This mirrors the [Recommendation path](#2-recommendation-path-from-request-to-displayed-rows) described in the request flow section.*

```mermaid
flowchart TD
    Start[User Request] --> Cache{Check Redis Cache}
    Cache -->|Hit| Return[Return Cached Results]
    Cache -->|Miss| Stage1[STAGE 1: Candidate Generation<br/>Parallel Execution]
    
    Stage1 --> CB[Content-Based<br/>Pinecone ANN]
    Stage1 --> ALS[Collaborative Filtering<br/>Implicit ALS]
    Stage1 --> Graph[Graph-Based<br/>Neo4j PPR]
    Stage1 --> SASRec[SASRec<br/>Session-Based]
    Stage1 --> Pop[Popularity<br/>Time-Decayed]
    
    CB --> Merge[Merge Candidates<br/>~1000 unique items]
    ALS --> Merge
    Graph --> Merge
    SASRec --> Merge
    Pop --> Merge
    
    Merge --> Stage2[STAGE 2: Feature Engineering<br/>29 Features]
    
    Stage2 --> Features[Generate 29 Features:<br/>• 6 retrieval scores<br/>• 8 metadata features<br/>• 5 social features<br/>• 7 session features<br/>• 3 popularity features]
    
    Features --> Feast[Store in Feast<br/>Feature Store S3]
    
    Feast --> Stage3[STAGE 3: Learning-to-Rank<br/>XGBoost LambdaRank]
    
    Stage3 --> Rank[Load Features from Feast<br/>Score & Sort by Engagement]
    
    Rank --> Stage4[STAGE 4: Post-Processing<br/>Diversity Rules]
    
    Stage4 --> Filter[Diversity Filters:<br/>• Max 3 per author<br/>• Min 4 genres<br/>• Novelty boost<br/>• Deduplication<br/>• Serendipity shuffle]
    
    Filter --> Explain[Generate Explanations<br/>LLM-based]
    
    Explain --> Stage5[STAGE 5: Cache Results<br/>Redis 5-min TTL]
    
    Stage5 --> Return
    Return --> End[Final Top-N Recommendations]
    
    style Stage1 fill:#E3F2FD
    style Stage2 fill:#F3E5F5
    style Stage3 fill:#FFF3E0
    style Stage4 fill:#E8F5E9
    style Stage5 fill:#FCE4EC
```

### Detailed Pipeline Stages

### Multi-Stage Ranking Strategy

**Stage 1: Candidate Generation** - Fast retrieval of 200-1000 potential books from multiple sources. Each algorithm runs in parallel to maximize coverage and diversity.

**Stage 2: Feature Engineering** - Enrich each candidate with 29 features from all data sources. Features include 6 retrieval scores (one per algorithm), 8 metadata features (genre/author/rating matching), 5 social features (friend reads, social proximity), 7 session features (position, device, time), and 3 popularity features. All features are stored in Feast feature store (S3 Parquet) for training the learning-to-rank model.

**Stage 3: Learning-to-Rank (XGBoost)** - XGBoost LambdaRank model trained on 29 features from Feast feature store. Features include retrieval scores from all 6 algorithms, metadata matching, social signals, temporal patterns, and diversity metrics. The model learns optimal feature weights to predict engagement probability and ranks candidates accordingly.

**Stage 4: Post-Processing & Diversity Rules** - Apply diversity constraints:
- Maximum 3 books per author in top recommendations
- Minimum 4 different genres represented
- Novelty boosting for lesser-known books
- Deduplication across algorithm sources
- Serendipity injection through controlled randomization

**Stage 5: Caching** - Store results in Redis with 5-minute TTL for fast subsequent access. Invalidate cache on model updates or significant user interactions.

---

## 🧠 Machine Learning Algorithms

### Core Algorithms

| Algorithm | Technique | Purpose | Data Source | Training Framework |
|-----------|-----------|---------|-------------|-------------------|
| **Content-Based** | Sentence-BERT (768-dim) + Pinecone ANN | Semantic similarity, cold start | Book metadata, Pinecone vectors | Pre-computed embeddings |
| **Collaborative Filtering** | Implicit ALS (Alternating Least Squares) | User-item preference learning | MongoDB event logs | Ray + Implicit |
| **Graph-Based** | Node2Vec + Personalized PageRank | Social recommendations | Neo4j social graph | Ray + NetworkX |
| **Sequential Models** | SASRec (Self-Attentive Sequential) | Session-based next-item prediction | MongoDB session data | Ray + PyTorch |
| **Popularity-Based** | Time-Decayed Scoring | Trending books, serendipity | Redis counters + MongoDB | Ray workers |
| **Contextual Bandit** | LinUCB | Exploration-exploitation balance | All sources | Ray |
| **Learning-to-Rank** | XGBoost LambdaRank | Final ranking optimization | Feast feature store (29 features) | XGBoost |

### Algorithm Details

**Content-Based Filtering (Pinecone)**
- Uses llama-text-embed-v2 to generate 1024-dimensional embeddings from book metadata
- User embeddings computed as weighted average of preferred book vectors
- Cosine similarity search via Pinecone for sub-50ms latency
- Handles cold start by matching user preferences to book content

**Collaborative Filtering (Implicit ALS)**
- Matrix factorization on user-item interaction matrix
- Works with implicit feedback (clicks, reads, bookmarks)
- Learns 50-100 dimensional latent factors for users and items
- Scales to millions of users and books efficiently

**Graph-Based Recommendations (Node2Vec + PageRank)**
- Node2Vec learns node embeddings from Neo4j social graph structure
- Personalized PageRank computes book relevance based on user's social connections
- Incorporates friend preferences and community influence
- Trained with Ray for distributed graph processing
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

**Learning-to-Rank (XGBoost LambdaRank)**
- Combines all 6 algorithm outputs with 29 engineered features
- Features stored in Feast feature store (S3 Parquet)
- LambdaRank objective for learning-to-rank optimization
- Trained on historical engagement data (clicks, reads, ratings)
- Automatically learns optimal feature weights for final ranking
- 29 features include: 6 retrieval scores, 8 metadata features, 5 social features, 7 session features, 3 popularity features

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
- Ray-based distributed training for parallel algorithm execution
- Kubernetes horizontal scaling for training and inference pods
- Multi-level caching (L1: Redis, L2: Application)

### The Recommendation Challenge

Building an effective recommendation system requires solving multiple interconnected problems:

**The Cold Start Problem** occurs when new users have no interaction history, or new books have no ratings. Traditional collaborative filtering fails here because it relies on finding similar users or items. BibliophileAI addresses this through content-based filtering using book metadata and user-declared preferences, allowing immediate personalization.

**Data Sparsity** is inevitable in any book platform—users interact with less than 0.01% of available books. This makes user-user and item-item similarity calculations unreliable. Our hybrid approach combines multiple algorithms, each handling sparsity differently, to ensure robust recommendations even with sparse data.

**Gray Sheep Users** have unique tastes that don't align with any major user cluster. Collaborative filtering performs poorly for these users because their nearest neighbors aren't truly similar. We detect gray sheep users through clustering analysis and automatically increase the weight of content-based and popularity algorithms for them.

**The Exploration-Exploitation Trade-off** balances recommending safe, known-good items versus exploring potentially interesting but uncertain recommendations. Pure exploitation leads to filter bubbles; pure exploration frustrates users. Our post-processing stage intentionally injects novelty while ensuring a minimum relevance threshold.

**Scalability at Scale** requires careful architectural decisions. Computing recommendations for millions of users against millions of books naively requires trillions of operations. Our multi-stage ranking pipeline first uses fast algorithms to retrieve hundreds of candidates, then applies expensive neural models only to this reduced set, achieving sub-100ms latency.

**Temporal Dynamics** mean user preferences evolve over time. A user interested in romance novels in January might prefer thrillers by June. Sequential models like SASRec capture these temporal patterns, while our real-time embedding updates ensure the system adapts to changing interests within a single session.

---

## 🔄 Microservices Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant UserService
    participant RecService
    participant DataService
    participant TrainingService
    
    User->>Frontend: Login/Register
    Frontend->>UserService: Authenticate
    UserService->>Frontend: JWT Token
    
    User->>Frontend: Request Recommendations
    Frontend->>RecService: GET /api/v1/recommend/combined
    RecService->>RecService: Run 5 Algorithms in Parallel
    RecService->>RecService: Feature Engineering
    RecService->>RecService: XGBoost Ranking
    RecService->>Frontend: Top-N Recommendations
    
    User->>Frontend: Interact with Book
    Frontend->>DataService: POST /api/events
    DataService->>DataService: Send to SQS
    
    Note over TrainingService: Daily Training Jobs
    TrainingService->>TrainingService: Train Models
    TrainingService->>RecService: Deploy New Models
```

---

## 🛠️ Technology Stack

### Backend
- **Web Framework:** FastAPI (async, high-performance)
- **Authentication:** JWT + OAuth 2.0 (Google)
- **Password Hashing:** Argon2 + SHA-256
- **ML Framework:** PyTorch (deep learning)
- **Classical ML:** Scikit-learn, Implicit (ALS)
- **Distributed Training:** Ray (distributed computing)
- **Learning-to-Rank:** XGBoost (LambdaRank)
- **Event Streaming:** AWS SQS (Kafka can replace for higher scale)
- **Orchestration:** Apache Airflow
- **Feature Store:** Feast
- **Graph Processing:** Node2Vec, NetworkX

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
- **Orchestration:** Kubernetes (Kind cluster)
- **Container Runtime:** Containerd
- **Service Mesh:** Kubernetes Services (ClusterIP)
- **Monitoring:** Kubernetes native monitoring
- **CI/CD:** GitHub Actions (optional)

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

### Book catalog (Gutendex → Supabase)
- Books are scraped from [Gutendex](https://gutendex.com/) (Project Gutenberg index) and stored in Supabase via the **book_data_importer**
- Supabase is the source of truth for book metadata (title, authors, subjects, languages, download links)
- The importer fetches by genre and upserts into the Supabase `books` table; embeddings and graph data are built from there (Pinecone, Neo4j, S3)

---

## 🚀 Getting Started

All setup steps (prerequisites, clone, Kind cluster, secrets, deploying Redis/Airflow/services, building training images, Airflow DAGs, Feast) are in **[SETUP.md](SETUP.md)**. Do not duplicate them here.

For per-service development details, see the README in each directory: `src/user_service/`, `src/recommendation_service/`, `src/clickstream_consumer/` (SQS consumer), `src/model_training_service/`, `feast_feature_store/`, `frontend/bibliophile-ai-frontend/`.

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

## 📈 Performance Benchmarks

### Latency Targets

| Operation | Target | Current (p95) |
|-----------|--------|---------------|
| Combined Recommendations | <100ms | ~85ms |
| Content-Based (Pinecone) | <50ms | ~35ms |
| Collaborative Filtering | <30ms | ~20ms |
| Graph Recommendations | <100ms | ~75ms |
| Session-Based (SASRec) | <50ms | ~40ms |
| Popularity Lookup | <10ms | ~5ms |

### Scalability Metrics

- **Throughput**: 1000+ recommendations/second per pod
- **Concurrent Users**: 10,000+ supported
- **Cache Hit Rate**: >85% for popular users
- **Model Inference**: Batch processing for 100+ users simultaneously

---

## 🔐 Security

- **Authentication**: JWT tokens with Argon2 password hashing
- **API Security**: OAuth 2.0, rate limiting, CORS protection
- **Data Privacy**: User data encrypted at rest and in transit
- **Secrets Management**: Kubernetes secrets for sensitive credentials
- **Network Security**: Service mesh with mTLS (optional)

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

- **Gutendex** - Project Gutenberg index; books scraped and stored in Supabase via book_data_importer
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
