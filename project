# ============================================================
# Behavioral Hybrid Recommender
# (Graph + ALS + Popularity)
# ============================================================

import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm

from node2vec import Node2Vec
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import faiss


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"

USER_EVENTS_PATH = os.path.join(DATA_DIR, "user_event_data.csv")
ITEMS_PATH = os.path.join(DATA_DIR, "item_information.csv")


# ============================================================
# LOAD DATA
# ============================================================

events = pd.read_csv(USER_EVENTS_PATH)
items = pd.read_csv(ITEMS_PATH)

print("Events shape:", events.shape)
print("Items shape:", items.shape)


# ============================================================
# EVENT WEIGHTING
# ============================================================

event_weights = {
    "view": 1,
    "click": 2,
    "favorite": 4,
    "purchase": 6
}

events["event_weight"] = (
    events["event_type"]
    .map(event_weights)
    .fillna(1)
    .astype(float)
)


# ============================================================
# USER–ITEM GRAPH
# ============================================================

G = nx.Graph()

agg = (
    events
    .groupby(["client_id", "item_id"])["event_weight"]
    .sum()
    .reset_index()
)

for _, r in agg.iterrows():

    G.add_edge(
        str(r["client_id"]),
        str(r["item_id"]),
        weight=float(r["event_weight"])
    )

print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())


# ============================================================
# NODE2VEC EMBEDDINGS
# ============================================================

node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=5,
    num_walks=10,
    p=1,
    q=1,
    workers=2
)

w2v = node2vec.fit(
    window=5,
    min_count=1,
    batch_words=128,
    workers=1,
    epochs=1
)

node_embeddings = {
    n: w2v.wv[n] if n in w2v.wv else np.zeros(64)
    for n in G.nodes()
}


# ============================================================
# ITEM INDEXING
# ============================================================

item_ids = items["item_id"].astype(str).tolist()

itemid_to_idx = {iid: i for i, iid in enumerate(item_ids)}
idx_to_itemid = {i: iid for iid, i in itemid_to_idx.items()}


# ============================================================
# GRAPH ITEM EMBEDDINGS
# ============================================================

graph_item_emb = np.zeros((len(item_ids), 64))

for i, iid in enumerate(item_ids):

    graph_item_emb[i] = node_embeddings.get(iid, np.zeros(64))

graph_item_emb = normalize(graph_item_emb)


# ============================================================
# COLLABORATIVE FILTERING (ALS)
# ============================================================

pivot = pd.pivot_table(
    events,
    index="client_id",
    columns="item_id",
    values="event_weight",
    aggfunc="sum",
    fill_value=0
)

als = AlternatingLeastSquares(
    factors=50,
    iterations=15,
    regularization=0.1
)

als.fit(csr_matrix(pivot.values.astype("float32")))


cf_item_emb = np.zeros((len(item_ids), 50))
col_map = {str(c): i for i, c in enumerate(pivot.columns)}

for i, iid in enumerate(item_ids):

    if iid in col_map:
        cf_item_emb[i] = als.item_factors[col_map[iid]]

cf_item_emb = normalize(cf_item_emb)


# ============================================================
# HYBRID EMBEDDING
# ============================================================

final_embeddings = normalize(
    np.hstack([graph_item_emb, cf_item_emb])
).astype("float32")


# ============================================================
# FAISS INDEX
# ============================================================

index = faiss.IndexFlatIP(final_embeddings.shape[1])
index.add(final_embeddings)


# ============================================================
# POPULARITY SIGNAL
# ============================================================

popularity = (
    events["item_id"]
    .astype(str)
    .value_counts()
    .to_dict()
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(seed_item_id, topk=10, expand_factor=5):

    seed_item_id = str(seed_item_id)

    if seed_item_id not in itemid_to_idx:
        raise ValueError("Seed item not found")

    qidx = itemid_to_idx[seed_item_id]
    qvec = final_embeddings[qidx:qidx + 1]

    D, I = index.search(qvec, topk * expand_factor)

    scored = []

    for idx, sim in zip(I[0], D[0]):

        if idx == qidx:
            continue

        iid = idx_to_itemid[idx]

        pop = np.log1p(popularity.get(iid, 0))

        score = sim * pop

        scored.append((iid, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:topk]
