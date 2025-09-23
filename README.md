# 📚 BibliophileAI: Social & Intelligent Book Recommendation System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

> **A scalable, microservices-based platform for personalized book recommendations using ensemble machine learning, graph algorithms, and real-time streaming infrastructure.**

---

## 🎯 Overview

BibliophileAI is a production-ready recommendation system that combines classical collaborative filtering, deep learning, and graph neural networks to deliver personalized book recommendations. Built with modern microservices architecture, it handles real-time user interactions, social features, and intelligent cold-start scenarios using data scraped from Google Books API.

### ✨ Key Highlights
- 🤖 **Multi-Model Ensemble**: Combines Content based Filtering, Collabrating filtering using Matrix Factorization, Deep lering Techniques using Neural Collabrative Filtering, and Graph Neural Networks for social-aware recommendation
- 🚀 **Real-Time Processing**: Event-driven architecture with Apache Kafka
- 🔍 **Vector Search**: Pinecone integration for semantic similarity and cold-start recommendations
- 🏗️ **Scalable Architecture**: Kubernetes-native with horizontal scaling capabilities
- 📊 **Social Features**: User preferences, ratings, and community-based recommendations
- 📚 **Rich Data Source**: Google Books API integration for comprehensive book metadata

---

## 🏗️ System Architecture

<div align="center">
  <img width="100%" alt="BibliophileAI Architecture Diagram" src="https://github.com/user-attachments/assets/bb2d2a75-ab76-4d4d-a76c-a616e24322a8" />
</div>

---

## 🔧 Microservices Architecture

### 👤 **User Service**
**Purpose**: Handles user authentication, profile management, and preferences

**Key Features:**
- JWT-based authentication with secure token management
- Google OAuth integration for seamless user onboarding
- User preference collection (favorite genres, reading habits)
- Profile management and user settings
- Session management and security

**Technologies:** FastAPI, Supabase (PostgreSQL), JWT, Google OAuth

---

### 📖 **Recommendation Service**
**Purpose**: Core ML engine that generates personalized book recommendations using industry-grade algorithms

**Key Features:**
- **Multi-Model Ensemble**: Combines collaborative filtering, content-based, and deep learning models
- **Real-Time Inference**: Sub-100ms latency recommendation generation
- **Cold Start Handling**: Content-based filtering using Pinecone vector embeddings
- **Social-Aware Recommendations**: Integrates Neo4j graph embeddings for social influence
- **Explanation Generation**: Provides interpretable reasoning for recommendations
- **A/B Testing**: Supports multiple model variants for continuous optimization
- **Gray Sheep Handling**: Clustering and hybrid methods for users with unique tastes

**ML Models:**
- **Content-Based Filtering**: Item metadata utilization with vector embeddings in Pinecone
- **Collaborative Filtering**: Matrix Factorization (SVD, NMF) for user-item interactions
- **Neural Collaborative Filtering (NCF)**: Deep learning for complex user behavior patterns
- **Social Recommendations**: Graph embeddings and social-aware methods using Neo4j
- **Hybrid Ensemble**: Dynamic combination of all models for final recommendations

**Technologies:** PyTorch, Scikit-learn, TorchServe, Neo4j Graph Data Science, Pinecone, Redis

---

### 📊 **Data Ingestion Service**
**Purpose**: Processes and streams user interaction events in real-time for model training and inference

**Key Features:**
- **Event Collection**: Captures clicks, ratings, reviews, reading progress, and social interactions
- **Data Validation**: Ensures data quality, schema compliance, and anomaly detection
- **Stream Processing**: Real-time event processing with Apache Kafka
- **Batch Processing**: Historical data analysis and bulk updates
- **Multi-Database Routing**: Distributes data to appropriate storage systems

**Event Types:**
- User interactions (clicks, views, time spent, reading progress)
- Ratings and reviews submission with sentiment analysis
- Social activities (follows, shares, recommendations)
- Search queries, filters, and browsing patterns
- Book metadata updates from Google Books API

**Technologies:** Apache Kafka, Apache Spark Streaming, Python, FastAPI

---

### 🔧 **Feature Engineering Service**
**Purpose**: Transforms raw data into ML-ready features for training and inference

**Key Features:**
- **User Feature Generation**: Demographics, reading history, social connections, temporal patterns
- **Item Feature Extraction**: Book metadata embeddings, genre encoding, popularity metrics
- **Contextual Features**: Time-based patterns, seasonal trends, current events correlation
- **Social Features**: Graph-based features from Neo4j (centrality, community clusters, influence scores)
- **Temporal Features**: Sequential patterns, reading velocity, preference evolution
- **Feature Store Management**: Centralized feature storage with versioning and lineage

**Technologies:** Apache Spark, Pandas, Scikit-learn, Neo4j Graph Data Science, Feature Store (Feast)

---

### 🤖 **Model Training Service**
**Purpose**: Automated ML pipeline for training, validation, and deployment of recommendation models

**Key Features:**
- **Scheduled Training**: Regular model retraining with fresh interaction data
- **Hyperparameter Optimization**: Automated tuning using Optuna or Ray Tune
- **Model Validation**: Cross-validation, A/B testing, and offline evaluation metrics
- **Experiment Tracking**: MLOps with model versioning, lineage, and performance monitoring
- **Multi-Algorithm Training**: Simultaneous training of multiple model types
- **Social Graph Training**: Graph neural network training using Neo4j data

**Training Pipeline:**
1. Data extraction from databases and feature store
2. Multi-model training (Content-based, CF, NCF, Social GNN)
3. Hyperparameter tuning for each algorithm
4. Model validation using precision@K, recall@K, NDCG metrics
5. Ensemble weight optimization
6. Model registration and deployment

**Technologies:** Apache Airflow, PyTorch, Scikit-learn, MLflow, Optuna, Neo4j, S3

---

### 📡 **Model Deployment Service**
**Purpose**: Serves trained ML models with high availability, scalability, and low latency

**Key Features:**
- **TorchServe Integration**: Production-grade PyTorch model serving with batching
- **Auto-scaling**: Kubernetes horizontal pod autoscaling based on traffic
- **Model Versioning**: Blue-green deployment with instant rollback capabilities
- **Load Balancing**: Intelligent request routing across model replicas
- **Health Monitoring**: Real-time model performance and drift detection
- **Multi-Model Serving**: Concurrent serving of different algorithm variants

**Supported Models:**
- **Content-Based Filtering**: Vector similarity using Pinecone embeddings
- **Matrix Factorization**: SVD and NMF collaborative filtering models
- **Neural Collaborative Filtering (NCF)**: Deep learning recommendation models
- **Graph Neural Networks (GNN)**: Social recommendation models from Neo4j
- **Hybrid Ensemble**: Combined prediction from all models

**API Endpoints:**
- `/recommend/content` - Content-based recommendations
- `/recommend/collaborative` - Matrix factorization recommendations
- `/recommend/neural` - Neural collaborative filtering
- `/recommend/social` - Graph-based social recommendations
- `/recommend/hybrid` - Final ensemble recommendations

**Technologies:** TorchServe, Kubernetes, Docker, Redis, Prometheus, Grafana

---

## 💾 Data Architecture

### 🗄️ **Database Strategy**

| Database | Purpose | Data Types |
|----------|---------|------------|
| **Supabase (PostgreSQL)** | Primary transactional data | User profiles, authentication, ratings, reviews |
| **MongoDB** | Book metadata and content | Book details from Google Books API, descriptions, metadata |
| **Neo4j** | Social connections & relationships | User-user connections, social graphs, community clusters |
| **Redis** | Caching and sessions | Recommendation cache, user sessions, rate limiting |
| **Pinecone** | Vector search | Book embeddings, similarity indices for content-based filtering |
| **S3** | Model artifacts | Trained models, feature stores, batch processing results |

### 📊 **Social Graph Service (Neo4j)**
**Purpose**: Manages social connections and relationship-based recommendations using graph database

**Key Features:**
- **Social Graph Management**: User-user connections, follows, and friendship networks
- **Community Detection**: Identify user clusters and interest groups using Louvain algorithm
- **Graph Embeddings**: Generate user and book embeddings from social graph structure
- **Social-Aware Recommendations**: Leverage friend preferences and social influence
- **Relationship Analytics**: Analyze social patterns and recommendation pathways

**Graph Structure:**
- **Nodes**: Users, Books, Genres, Authors
- **Relationships**: FOLLOWS, RATED, LIKES, SIMILAR_TO, BELONGS_TO
- **Properties**: Connection strength, interaction timestamps, preference weights

**Social Algorithms:**
- Graph-based collaborative filtering
- Social influence modeling
- Community detection (Louvain, Label Propagation)
- Personalized PageRank for recommendation ranking

**Technologies:** Neo4j, Cypher queries, Graph Data Science library

---

## 🔄 Data Flow

### 📈 **End-to-End Data Pipeline**

1. **Data Ingestion**: Google Books API → Data processing → MongoDB (book storage)
2. **User Interactions**: Frontend → API Gateway → Event streaming (Kafka)
3. **Social Connections**: User relationships → Neo4j → Graph-based recommendations
4. **Real-Time Processing**: Kafka → Data Ingestion Service → Multi-database updates
5. **Feature Engineering**: Raw data → Feature extraction → Feature store → ML models
6. **Model Training**: Batch processing → Multi-algorithm training → Model registry → S3 storage
7. **Inference Pipeline**: User request → Recommendation Service → Multi-model ensemble → Final recommendations
8. **Feedback Loop**: User interactions → Model retraining → Improved recommendations

### 🚀 **Real-Time Data Flow**
- **User Action** → **Kafka Event** → **Feature Update** → **Real-time Recommendation Update**
- **Social Activity** → **Neo4j Update** → **Graph Embedding Refresh** → **Social Recommendation Update**
- **New Book Data** → **MongoDB Storage** → **Pinecone Embedding** → **Content-based Recommendations**

---

## 🤖 Advanced Machine Learning Pipeline

### 🧠 **Industry-Grade Recommendation Algorithms**

Our recommendation system combines multiple state-of-the-art techniques in a layered, hybrid architecture comparable to Netflix and Instagram:

| **Aspect** | **Technique/Algorithm** | **Purpose/Benefit** |
|------------|------------------------|---------------------|
| **Content-Based Filtering** | Item metadata utilization (Pinecone vector embeddings) | Recommends similar content; handles cold start for new users |
| **Collaborative Filtering** | Matrix Factorization (SVD, NMF) | Captures latent user preferences; accurate rating prediction |
| **Hybrid Recommendation** | Combination of CF + Content | Robust to sparse data and cold start issues |
| **Deep Learning** | Neural Collaborative Filtering, RNN/Transformer models | Captures complex, temporal user behavior patterns |
| **Social Recommendations** | Graph embeddings in Neo4j, social-aware methods | Incorporates social influence in recommendations |
| **Cold Start Handling** | Content-based filtering using Pinecone | Improves recommendations for new users/items |
| **Gray Sheep Problem** | Clustering, hybrid methods | Handles users with unique tastes effectively |
| **Scalability & Real-Time** | Approximate nearest neighbors, streaming data, multi-stage ranking | Ensures low latency and scale for millions of users |

### 🎯 **Algorithm Implementation Details**

**1. Content-Based Filtering (Pinecone)**
- **Book Embeddings**: Transformer-based embeddings of book metadata and descriptions
- **Vector Similarity**: Cosine similarity search in Pinecone vector database
- **Cold Start Solution**: Immediate recommendations for new users based on preferences
- **Semantic Search**: Natural language query matching with book content

**2. Collaborative Filtering (Matrix Factorization)**
- **Singular Value Decomposition (SVD)**: Captures latent factors in user-item interactions
- **Non-Negative Matrix Factorization (NMF)**: Interpretable latent factors
- **Implicit Feedback**: Handles clicks, views, and reading time data
- **Temporal Factors**: Time-aware matrix factorization for evolving preferences

**3. Neural Collaborative Filtering (NCF)**
- **Deep Neural Networks**: Multi-layer perceptrons for complex user-item interactions
- **Embedding Layers**: Dense representations of users and items
- **Non-linear Modeling**: Captures complex patterns beyond traditional matrix factorization
- **Attention Mechanisms**: Focus on relevant interaction patterns

**4. Social Recommendations (Neo4j Graph)**
- **Graph Neural Networks**: Node embeddings from social graph structure
- **Social Matrix Factorization**: Incorporate social regularization in CF
- **Community-Based Filtering**: Recommendations from user communities
- **Trust Propagation**: Model trust relationships in recommendation scoring

**5. Hybrid Ensemble Strategy**
- **Dynamic Weighting**: Context-aware combination of model predictions
- **Stacking Approach**: Meta-learning for optimal model combination
- **Multi-Objective Optimization**: Balance accuracy, diversity, and novelty
- **Contextual Bandits**: Adaptive model selection based on user context

### 🧩 **Specialized Problem Solving**

**Cold Start Handling:**
- **New Users**: Content-based recommendations using demographic info and genre preferences
- **New Books**: Content similarity using Pinecone embeddings and metadata
- **Active Learning**: Strategic questioning to quickly learn user preferences

**Gray Sheep Problem:**
- **User Clustering**: Identify users with unique taste patterns
- **Hybrid Fallback**: Use content-based when collaborative filtering fails
- **Outlier Detection**: Special handling for users with rare preferences

**Scalability Optimization:**
- **Approximate Nearest Neighbors**: FAISS integration with Pinecone for fast similarity search
- **Multi-Stage Ranking**: Candidate generation → Feature scoring → Final ranking
- **Caching Strategy**: Multi-level caching with Redis for hot recommendations
- **Batch Processing**: Efficient batch inference for multiple users

---

## ⚡ Key Features

### 🔐 **Authentication & User Management**
- Secure JWT-based authentication with refresh tokens
- Google OAuth integration for frictionless sign-up
- User preference collection and management
- Profile customization and reading history tracking

### 📊 **Intelligent Recommendations**
- **Personalized**: Multi-algorithm ensemble tailored to individual users
- **Social**: Community-based recommendations from Neo4j social graph
- **Contextual**: Time-aware and seasonal recommendations
- **Explainable**: Clear reasoning for why books are recommended
- **Diverse**: Balanced recommendations across genres and authors using hybrid methods

### 🔍 **Advanced Search & Discovery**
- **Semantic Search**: Natural language book discovery using Pinecone
- **Filter & Sort**: Advanced filtering by genre, author, rating, publication date
- **Similar Books**: Content-based similarity using vector embeddings
- **Social Discovery**: Friend recommendations and community trends

---

## 🛠️ Technology Stack

<details>
<summary><b>🔧 Backend Technologies</b></summary>

- **Python 3.9+** - Core programming language
- **FastAPI** - High-performance web framework
- **PyTorch** - Deep learning and neural networks
- **Scikit-learn** - Classical ML algorithms
- **TorchServe** - Production model serving
- **Apache Kafka** - Event streaming platform
- **Apache Airflow** - Workflow orchestration
- **Apache Spark** - Big data processing

</details>

<details>
<summary><b>💾 Data & Storage</b></summary>

- **Supabase (PostgreSQL)** - Cloud-native PostgreSQL for scalable user data
- **MongoDB** - Document storage for book metadata from Google Books API
- **Neo4j** - Graph database for social connections and relationships
- **Redis** - Caching and session storage
- **Pinecone** - Vector database for similarity search
- **AWS S3** - Model artifacts and data lake storage

</details>

<details>
<summary><b>🎨 Frontend Technologies</b></summary>

- **React 18** - Modern UI library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **React Query** - Server state management
- **React Router** - Client-side routing

</details>

<details>
<summary><b>🏗️ Infrastructure</b></summary>

- **Docker** - Containerization
- **Kubernetes** - Container orchestration
- **Prometheus** - Metrics and monitoring
- **Grafana** - Data visualization dashboards
- **MLflow** - MLOps and experiment tracking

</details>

---

## 📊 Monitoring & Analytics

### 📈 **Key Metrics**
- **Recommendation Quality**: Precision@K, Recall@K, NDCG, Diversity, Coverage
- **User Engagement**: Click-through rates, session duration, return visits
- **System Performance**: Response latency, throughput, error rates
- **Business KPIs**: User retention, recommendation coverage, catalog discovery

### 🔍 **Observability**
- **Real-Time Monitoring**: Prometheus + Grafana dashboards
- **Model Performance**: MLflow tracking and drift detection
- **Distributed Tracing**: Request flow across microservices
- **Log Aggregation**: Centralized logging with structured data

---

## 🤝 Contributing

We welcome contributions! This project is actively being developed with the following areas open for contribution:

### 🛠️ **Current Development Areas**
- Machine learning model improvements
- Social graph algorithm enhancements
- Performance optimizations
- Additional recommendation algorithms
- Frontend UI/UX improvements

### 📝 **How to Contribute**
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Submit** a pull request with detailed description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Books API** - Comprehensive book metadata and information
- **PyTorch Ecosystem** - Deep learning framework and community
- **Neo4j** - Graph database platform for social recommendations
- **Pinecone** - Vector database platform for ML applications
- **Open Source Community** - Countless libraries and tools that make this possible

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/RAAHUL-tech/BibliophileAI?style=social)](https://github.com/RAAHUL-tech/BibliophileAI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/RAAHUL-tech/BibliophileAI?style=social)](https://github.com/RAAHUL-tech/BibliophileAI/network/members)

---

**Built with ❤️ by [Raahul Krishna Durairaju](https://github.com/RAAHUL-tech)**

</div>
