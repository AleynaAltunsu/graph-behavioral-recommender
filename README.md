# Behavioral Hybrid Recommender System
(Graph + Collaborative Filtering + Popularity)

This project implements a **pure behavioral hybrid recommendation system**
designed for real estate listing recommendation.

The system combines:

• Graph-based embeddings (Node2Vec)
• Collaborative Filtering (ALS)
• Popularity-based ranking
• FAISS similarity search

The goal is to recommend similar real estate listings based on user behavior signals.

## Running the Project

git clone https://github.com/AleynaAltunsu/graph-behavioral-recommender

cd graph-behavioral-recommender

pip install -r requirements.txt

python src/recommender.py


recommend(seed_item_id="12345", topk=10)
---

# Problem

Traditional recommendation approaches struggle when:

• user behavior is sparse
• items have limited metadata
• cold-start interactions occur

To address this, we model the **entire interaction space as a graph** and learn latent representations.

---

# System Architecture

User Behavior → Graph Construction → Node2Vec Embedding
                                 ↓
                        Collaborative Filtering (ALS)
                                 ↓
                         Hybrid Embedding Space
                                 ↓
                           FAISS Index
                                 ↓
                         Top-N Recommendations

---

# Data

Two datasets are used:

### User Events

Contains user interactions.

Example fields:

client_id  
item_id  
event_type (view, click, favorite, purchase)

### Item Information

Contains listing metadata such as:

item_id  
title / listing name  

Sensitive or proprietary fields have been removed.

---

# Event Weighting

User actions are weighted to represent behavioral strength:

| Event | Weight |
|------|------|
view | 1
click | 2
favorite | 4
purchase | 6

This produces a stronger signal for high-intent interactions.

---

# Graph Representation

A bipartite graph is created:

Users ↔ Items

Edges represent interactions weighted by event strength.

Node2Vec learns behavioral proximity between nodes.

---

# Collaborative Filtering

Matrix Factorization is applied using **ALS**.

This captures:

• implicit user preference  
• latent item similarity

---

# Hybrid Representation

Final item vectors are produced by concatenating:

Graph Embeddings (Node2Vec)  
Collaborative Embeddings (ALS)

Final dimension:

114

These vectors represent items in a unified behavioral space.

---

# Retrieval Layer

Similarity search is implemented using **FAISS**.

This enables efficient nearest neighbor retrieval even for large catalogs.

---

# Ranking

Recommendations are ranked using:

score = similarity × log(popularity)

This balances:

• behavioral similarity  
• item popularity

---

# Running the Project

Clone repository:
