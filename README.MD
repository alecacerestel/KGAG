# KGAG: Knowledge-Aware Group Representation Learning

This module implements the **KGAG (Knowledge-Aware Group Attention Graph)** model using PyTorch. [cite_start]It is based on the research paper *"Knowledge-Aware Group Representation Learning for Group Recommendation"* (ICDE 2021)[cite: 3].

## What is Group Recommendation?

Unlike traditional recommendation systems that suggest items to a single user, **Group Recommendation** aims to suggest items (e.g., movies, restaurants) that satisfy a group of users collectively.

### The Challenge
This implementation focuses on **Occasional Groups** (ad-hoc groups formed temporarily, e.g., friends going to a cinema). [cite_start]These groups suffer from severe **data sparsity** because they rarely interact with items as a specific unit[cite: 7, 27].

### The KGAG Solution
To solve this, KGAG does not rely solely on historical group interactions. Instead, it uses a **Knowledge Graph (KG)** to:
1.  [cite_start]**Connect Users via Interests:** If User A likes "Movie X" and User B likes "Movie Y", and both movies share the same Director in the KG, the model understands these users share a latent interest[cite: 48].
2.  [cite_start]**Model Influence:** It calculates which user has the most "say" in the group decision based on the specific item being considered[cite: 11].

---

## Architecture & Phases

The implementation is modularized into 4 distinct phases corresponding to the paper's architecture:

### Phase 1: Raw Data & Embeddings (`model.py`)
We initialize **Zero-order Representations** for:
* **Users**
* **Entities** (Items + KG entities like Directors, Genres)
* **Relations** (Semantic connections in the KG)

### Phase 2: Information Propagation (`layers.py`)
* **Concept:** Collaborative Knowledge Graph (CKG) Construction.
* **Mechanism:** We construct a heterogeneous graph merging User-Item interactions with Knowledge Graph edges.
* **Logic:** A **Graph Convolution Network (GCN)** recursively propagates information.
    * [cite_start]*Neighbor Aggregation:* Aggregates features from connected entities[cite: 206].
    * [cite_start]*High-Order Propagation:* Stacks multiple layers so users can "see" distant connections in the graph[cite: 240].

### Phase 3: Preference Aggregation (Attention Mechanism)
This is the core decision-making logic (currently being implemented). It determines the group's final vector $g$ by aggregating members $u$ based on two factors:
1.  [cite_start]**Self Persistence (SP):** How much does User $u$ personally like the candidate item? [cite: 274]
2.  [cite_start]**Peer Influence (PI):** How much does the rest of the group agree with User $u$? [cite: 278]

### Phase 4: Optimization
* **Loss Function:** Pairwise Margin Loss.
* [cite_start]**Goal:** Ensures the score of a positive item (selected by the group) is higher than a negative item by a specific margin $M$[cite: 13].

---