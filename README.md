# BibliophileAI: Social & Intelligent Book Recommendation System

A scalable, microservices-based platform for personalized book recommendations using ensemble machine learning, graph algorithms, and real-time streaming. Built on Kubernetes, Python, React, and all major modern data infrastructure.

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project delivers personalized book recommendations powered by classical, neural, and graph-based ML approaches using a rich dataset with social network and rating data.

---

## Project Architecture

- **Frontend:** React SPA for user interactions
- **Backend:** FastAPI-based microservices
- **Model Serving:** TorchServe, custom APIs
- **Databases:** PostgreSQL, MongoDB, Redis
- **Messaging:** Apache Kafka
- **Workflow:** Apache Airflow for data pipelines
- **Vector Search:** Pinecone for similarity-based cold start recommendations
- **Orchestration:** Kubernetes (via Docker Desktop/Minikube)
- **Monitoring/Logging:** Prometheus, Grafana, ELK

---

## Features

- User authentication and profile management
- Ensemble recommendation engine (CF, MF, NCF, GNN)
- Real-time event ingestion (ratings, clicks, reviews)
- Batch and streaming model retraining
- Content-based filtering for cold start users
- Rich interactive frontend and dashboards

---

## Technology Stack

- Python 3.8+, FastAPI, PyTorch, Airflow, Kafka, Pinecone
- Docker, Kubernetes, Helm
- PostgreSQL, MongoDB, Redis
- Node.js, React
- Prometheus, Grafana

---
## Architecture
<img width="2582" height="1224" alt="image" src="https://github.com/user-attachments/assets/b7a5677d-40d4-4faf-8653-c1d7e6355c37" />

---

## Installation
Clone the repository
`` git clone https://github.com/RAAHUL-tech/BibliophileAI.git ``
Backend setup (Python)
`` cd src/
pip install -r requirements.txt ``

Frontend setup (React)
`` cd frontend/
npm install
npm start ``

Infrastructure setup (Kubernetes manifests)
`` kubectl apply -f kubernetes/ ``

---

## Usage

- Visit the frontend and register/login as a user
- Interact with book catalog and provide ratings
- Receive real-time recommendations and feedback loops
- View logs/metrics in Grafana dashboards

---

## Project Structure

See [folder structure](#project-structure) in the documentation for all directories and files.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss the proposed change.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

