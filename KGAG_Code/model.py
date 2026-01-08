"""
KGAG Model: Knowledge-Aware Group Representation Learning for Group Recommendation

================================================================================
PAPER FOUNDATIONS (Section I - Introduction):
================================================================================

PROBLEM DEFINITION:
- Target: Occasional group recommendation
- Challenge 1: Data sparsity due to lack of group-item interaction records
- Challenge 2: Aggregating member preferences for group decision making

KEY CONTRIBUTIONS (Section I):
1. Knowledge Graph Integration:
   "Our KGAG model integrates the group recommendation with knowledge graph
    in which there are abundant structure and semantic information of items
    helpful for handling the sparsity problem" (Section I, contribution 1)

2. Attention Mechanism:
   "We propose an attention mechanism that can learn the influence of each
    user in a group to better model the group decision-making process. We take
    both user-user interaction and the influence of candidate item in group
    into consideration" (Section I, contribution 2)

3. Margin Loss:
   "To learn more discriminative representations of groups and items, we also
    extend the idea of margin loss which not only requires the score of the
    positive sample to be higher than that of the negative sample, but also
    requires the score of the positive sample is higher than that of the
    negative sample by a given margin" (Section I, contribution 3)

================================================================================
ARCHITECTURE OVERVIEW (from Figure 1 and Section I):
================================================================================

Phase 1: Raw Data Inputs
  - Knowledge Graph G = (E, R): entities and relations
    Example triple: (Alfred Hitchcock, DirectorOf, Psycho)
  - User-Item interactions Y^u
  - Group-Item interactions Y^g

Phase 2: Collaborative Knowledge Graph (CKG) Construction
  "each item can be mapped to an entity in knowledge graph by matching the name,
   so that recommendation system makes contact with the knowledge graph and gets
   its information." (Section I, para 4)
  
  CKG Structure:
    - Nodes: Users ∪ Entities (items are subset of entities)
    - Edges: User-Item interactions + KG relations
    - "GCN propagates node information along edges of knowledge graph, a node
       can get information from its neighborhood" (Section I, para 4)

Phase 3: Attention-based Group Aggregation
  "the influence of each member on the final result should be different and it
   varies according to other group members and candidate item." (Section I, para 6)
  
  Learns member influence based on:
    - User-user interaction (interest similarity via KG connectivity)
    - Candidate item (item-specific influence)

Phase 4: Model Optimization (Margin Loss)
  "we also extend the idea of margin loss which not only requires the score of
   the positive sample to be higher than that of the negative sample, but also
   requires the score of the positive sample is higher than that of the negative
   sample by a given margin" (Section I, contribution 3)

================================================================================
This model learns user and item representations by propagating information
through the Collaborative Knowledge Graph structure.
================================================================================
"""

import torch
import torch.nn as nn
from .layers import GCNLayer
from .aggregators import AttentionGroupAggregator


class KGAG(nn.Module):
    """
    Knowledge-Aware Group Recommendation Model
    
    Paper Reference: Section I - Introduction
    
    ============================================================================
    ARCHITECTURE OVERVIEW (from paper):
    ============================================================================
    
    Phase 1: Raw Data Inputs
      - Knowledge Graph G = (E, R): entities and relations
      - User-Item interactions Y^u
      - Group-Item interactions Y^g
    
    Phase 2: Collaborative Knowledge Graph (CKG) Construction
      Paper quote (Section I, para 4):
      "each item can be mapped to an entity in knowledge graph by matching
       the name, so that recommendation system makes contact with the
       knowledge graph and gets its information"
      
      CKG = User-Item edges + Knowledge Graph edges
        Users <-> Items (via interactions)
        Items <-> Entities (via KG relations)
      
      Paper quote (Section I, para 4):
      "a graph convolution network (GCN) is employed to capture abundant
       structure information of items and users in knowledge graph to
       overcome the sparsity problem"
      
      GCN propagates through this heterogeneous structure to generate
      knowledge-aware high-order representations.
    
    Phase 3: Attention Mechanism
      Paper quote (Section I, contribution 2):
      "We propose an attention mechanism that can learn the influence of
       each user in a group to better model the group decision-making process.
       We take both user-user interaction and the influence of candidate item
       in group into consideration"
      
      Key insight (Section I, for 5):
      "if Jack is interested in movie Psycho and Rose is interested in movie
       Rear Window, it may find that Jack and Rose are both interested in the
       movies which are directed by Alfred Hitchcock. Moreover, more high-order
       connectivities between two users imply the more similar interests the
       two users share"
    
    ============================================================================
    KEY INSIGHT (Section I, for 6):
    ============================================================================
    "the influence of each member on the final result should be different and
     it varies according to other group members and candidate item"
    
    Why not social networks? (Section I, for 6):
    "a person who is active in the group may not know much about movies. When
     this group wants to see a movie, she will also consult other members of
     the group who know better about movies. Therefore, she does not necessarily
     play a major role in this group's decision making"
    
    ============================================================================
    """
    
    def __init__(self, num_users, num_entities, num_relations, 
                 embedding_dim=64, num_layers=3, aggregation_type='bi-interaction'):
        """
        Initialize KGAG model
        
        ========================================================================
        PAPER FOUNDATION (Section I, for 4):
        ========================================================================
        "a GCN is leveraged to capture the structure and semantic information
         of items and the similarity between users in terms of interest due to
         its ability of propagating the information along with the connectivity
         of knowledge graph."
        
        
        INPUTS:
        
        Args:
            num_users (int): Total number of users U = {u1, u2, ..., u_N}
            
            num_entities (int): Total number of entities E = {e1, e2, ..., e_K}
                               Entities include:
                                 - Items V = {v1, v2, ..., v_M} (subset of E)
                                 - Auxiliary KG entities (e.g., directors, genres)
                               Paper example (Section I): Alfred Hitchcock is
                               an entity but not an item
            
            num_relations (int): Number of relation types R = {r1, r2, ..., r_L}
                                Relations define semantic connections in KG
                                Example (Section I): DirectorOf, ActorOf, etc.
                                Note: User-Item interaction can be treated as
                                a special relation type
            
            embedding_dim (int): Dimension d of embeddings (default 64)
                                e_u ∈ ℝ^d for users
                                e_i ∈ ℝ^d for entities
                                e_r ∈ ℝ^d for relations
            
            num_layers (int): Number of GCN layers L (default 3)
                             Controls propagation depth (high-order connectivity)
                             Paper insight (Section I, para 5):
                             "more high-order connectivities between two users
                              imply the more similar interests the two users share"
            
            aggregation_type (str): Type of GCN aggregation
                                   Options: 'bi-interaction', 'gcn', 'graphsage'
        
        ========================================================================
        INITIALIZATION STRATEGY:
        ========================================================================
        Zero-order representations (before CKG propagation):
          - e_u^(0): Initial user embedding
          - e_i^(0): Initial entity embedding
          - e_r^(0): Initial relation embedding
        
        """
        super(KGAG, self).__init__()
        
        self.num_users = num_users
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Total nodes in CKG = Users + Entities
        # CKG structure: [User_0, ..., User_N, Entity_0, ..., Entity_M]
        self.num_nodes_ckg = num_users + num_entities
        
        # Initialize embeddings (zero-order representations)
        # These will be refined through CKG propagation
        
        # User embeddings: e_u^(0) for u in Users
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )
        
        # Entity embeddings: e_i^(0) for i in Entities
        # Entities include both items and auxiliary KG entities
        # Items are typically the first subset of entities
        self.entity_embedding = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim
        )
        
        # Relation embeddings: e_r^(0) for r in Relations
        # Relations define semantic connections in the knowledge graph
        # Note: User-Item interaction can be treated as a special relation type
        self.relation_embedding = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim
        )
        
        # Initialize embeddings with Xavier uniform distribution
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        # Build GCN layers for CKG propagation (Phase 2)
        # These layers will operate on the Collaborative Knowledge Graph
        self.gcn_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            self.gcn_layers.append(
                GCNLayer(
                    in_dim=embedding_dim,
                    out_dim=embedding_dim,
                    aggregation_type=aggregation_type
                )
            )
        
        # Group aggregator with attention mechanism
        self.group_aggregator = AttentionGroupAggregator(
            embedding_dim=embedding_dim,
            attention_type='concat',
            dropout=0.1
        )
    
    def construct_ckg_adjacency(self, user_item_edges, kg_edges):
        """
        Phase 2: Construct Collaborative Knowledge Graph (CKG) adjacency structure
        
        ========================================================================
        PAPER FOUNDATION (Section I, for 4):
        ========================================================================
        "each item can be mapped to an entity in knowledge graph by matching
         the name, so that recommendation system makes contact with the
         knowledge graph and gets its information"
        
        The CKG merges two types of edges:
        1. User-Item interaction edges (collaborative signal from Y^u)
           → Captures user preferences and behavior patterns
        
        2. Knowledge Graph edges (semantic signal from KG)
           → Captures item attributes and inter-item relations
        
        ========================================================================
        CKG STRUCTURE:
        ========================================================================
        Nodes: [User_1, ..., User_N, Entity_1, ..., Entity_K]
              Total nodes = N + K
        
        Edges (bidirectional):
          - User u <-> Item v  (if user u interacted with item v)
          - Entity e1 <-> Entity e2  (if (e1, relation, e2) in KG)
        
        Example from paper (Section I, para 5):
          Jack -> Psycho -> Alfred Hitchcock <- Rear Window <- Rose
          This path shows Jack and Rose have similar interests
        
        ========================================================================
        NORMALIZATION (Laplacian):
        ========================================================================
        A_norm[i,j] = 1 / sqrt(deg(i) * deg(j))
        
        This normalization:
          - Prevents high-degree nodes from dominating propagation
          - Ensures stable gradient flow during training
          - Standard practice in GCN-based models
        
        ========================================================================
        INPUTS:
        ========================================================================
        Args:
            user_item_edges: Tensor of shape (2, num_ui_edges)
                            user_item_edges[0] = user indices (0 to N-1)
                            user_item_edges[1] = item indices (0 to M-1)
                            Represents user-item interactions Y^u
            
            kg_edges: Tensor of shape (2, num_kg_edges)
                     kg_edges[0] = head entity indices (0 to K-1)
                     kg_edges[1] = tail entity indices (0 to K-1)
                     Represents KG relations (ignoring relation types here)
        
        ========================================================================
        OUTPUT:
        ========================================================================
        Returns:
            adjacency_matrix: Sparse tensor of shape (N+K, N+K)
                             Normalized adjacency matrix of CKG
                             Used for GCN propagation in Phase 2
        
        ========================================================================
        """
        
        # Build edge list for CKG
        # Need to adjust indices since CKG has structure: [Users, Entities]
        
        # User-Item edges: map user index u to node index u
        #                  map item index i to node index (num_users + i)
        ui_src = user_item_edges[0]  # User indices
        ui_dst = user_item_edges[1] + self.num_users  # Item indices (offset by num_users)
        
        # Make user-item edges bidirectional (undirected graph)
        ui_edges = torch.cat([
            torch.stack([ui_src, ui_dst]),  # User -> Item
            torch.stack([ui_dst, ui_src])   # Item -> User
        ], dim=1)
        
        # Knowledge Graph edges: both head and tail are entities
        # Map entity indices to CKG node indices (offset by num_users)
        kg_src = kg_edges[0] + self.num_users
        kg_dst = kg_edges[1] + self.num_users
        
        # Make KG edges bidirectional
        kg_edges_bi = torch.cat([
            torch.stack([kg_src, kg_dst]),  # Head -> Tail
            torch.stack([kg_dst, kg_src])   # Tail -> Head
        ], dim=1)
        
        # Combine all edges into unified CKG edge list
        all_edges = torch.cat([ui_edges, kg_edges_bi], dim=1)
        
        # Build sparse adjacency matrix with Laplacian normalization
        # Normalization: A_norm[i,j] = 1 / sqrt(deg(i) * deg(j))
        
        num_edges = all_edges.shape[1]
        
        # Compute node degrees
        node_indices = all_edges.flatten()
        degree = torch.bincount(node_indices, minlength=self.num_nodes_ckg).float()
        
        # Compute normalization factor for each edge
        src_degree = degree[all_edges[0]]
        dst_degree = degree[all_edges[1]]
        edge_norm = 1.0 / torch.sqrt(src_degree * dst_degree + 1e-10)
        
        # Create sparse adjacency matrix
        indices = all_edges
        values = edge_norm
        adjacency_matrix = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(self.num_nodes_ckg, self.num_nodes_ckg)
        )
        
        return adjacency_matrix
    
    def forward(self, user_indices, item_indices, user_item_edges, kg_edges):
        """
        Forward pass: Propagate information through Collaborative Knowledge Graph.
        
        Phase 2 Implementation:
        1. Construct CKG from user-item interactions + knowledge graph
        2. Propagate embeddings through GCN layers on CKG
        3. Generate knowledge-aware high-order representations
        
        Implements High-order Information Propagation (Eq. 7 and Eq. 8):
        
        Eq. 7 (Layer-wise Propagation on CKG):
            e^(l+1) = GCN(e^(l), A_ckg)
            where A_ckg is the CKG adjacency matrix combining
            user-item edges and knowledge graph edges
        
        Eq. 8 (Multi-hop Aggregation):
            e_final = aggregate(e^(0), e^(1), ..., e^(L))
            Combines representations from all propagation layers
        
        Args:
            user_indices: User indices in batch, shape (batch_size,)
            item_indices: Item indices in batch, shape (batch_size,)
            user_item_edges: User-item interaction edges, shape (2, num_ui_edges)
            kg_edges: Knowledge graph edges, shape (2, num_kg_edges)
        
        Returns:
            user_embeddings: Knowledge-aware user representations, shape (batch_size, embedding_dim)
            item_embeddings: Knowledge-aware item representations, shape (batch_size, embedding_dim)
        """
        
        # Step 1: Build Collaborative Knowledge Graph (CKG) structure
        # This is the key innovation: merging collaborative and semantic signals
        ckg_adjacency = self.construct_ckg_adjacency(user_item_edges, kg_edges)
        
        # Step 2: Get zero-order embeddings (initial representations)
        all_user_embeddings = self.user_embedding.weight  # (num_users, embedding_dim)
        all_entity_embeddings = self.entity_embedding.weight  # (num_entities, embedding_dim)
        
        # Combine into unified CKG node embeddings
        # Node ordering: [User_0, ..., User_N, Entity_0, ..., Entity_M]
        ckg_embeddings = torch.cat([all_user_embeddings, all_entity_embeddings], dim=0)
        
        # Store embeddings from each layer for multi-hop aggregation (Eq. 8)
        embeddings_per_layer = [ckg_embeddings]
        
        # Step 3: High-order Information Propagation through CKG (Eq. 7)
        # Each layer aggregates information from neighbors in the CKG
        current_embeddings = ckg_embeddings
        
        for layer_idx, gcn_layer in enumerate(self.gcn_layers):
            # Propagate through CKG structure
            # e^(l+1) = GCN(e^(l), A_ckg)
            # This allows:
            #   - Users to learn from items they interacted with
            #   - Items to learn from related entities in KG
            #   - Information to flow: Users <-> Items <-> Entities
            current_embeddings = gcn_layer(
                ego_embeddings=current_embeddings,
                adjacency_matrix=ckg_adjacency
            )
            
            # Store layer output for multi-hop aggregation
            embeddings_per_layer.append(current_embeddings)
        
        # Step 4: Multi-hop Aggregation (Eq. 8)
        # Combine representations from all layers to capture different receptive fields
        # Layer 0: Direct connections
        # Layer 1: 1-hop neighbors
        # Layer 2: 2-hop neighbors, etc.
        final_embeddings = torch.stack(embeddings_per_layer, dim=0).sum(dim=0)
        
        # Step 5: Extract knowledge-aware embeddings for users and items
        # Split combined embeddings back into users and entities
        final_user_embeddings = final_embeddings[:self.num_users]
        final_entity_embeddings = final_embeddings[self.num_users:]
        
        # Get specific embeddings for the batch
        batch_user_embeddings = final_user_embeddings[user_indices]
        batch_item_embeddings = final_entity_embeddings[item_indices]
        
        return batch_user_embeddings, batch_item_embeddings
    
    def get_group_embedding(self, group_members, item_indices, user_item_edges, kg_edges):
        """
        Generate group embedding from member user embeddings using attention mechanism.
        
        This aggregates individual user representations into a single group representation
        using the attention-based aggregator that considers both member-member similarity
        and candidate item influence.
        
        Args:
            group_members: List of user indices for the group
                          Example: [0, 1, 2] or tensor of shape (num_members,)
            item_indices: Item indices to recommend, shape (batch_size,)
            user_item_edges: User-item interaction edges for CKG construction
            kg_edges: Knowledge graph edges for CKG construction
        
        Returns:
            group_embeddings: Aggregated embeddings for the group, shape (batch_size, embedding_dim)
            attention_weights: Attention weights for interpretability, shape (batch_size, num_members)
        """
        # Get knowledge-aware user and item embeddings
        if isinstance(group_members, list):
            group_members = torch.tensor(group_members, dtype=torch.long)
        
        # Get all embeddings after CKG propagation
        all_user_embeddings, all_item_embeddings = self.get_all_embeddings(
            user_item_edges, kg_edges
        )
        
        # Extract member embeddings
        member_embeddings = all_user_embeddings[group_members]  # (num_members, embedding_dim)
        
        # Extract item embeddings for batch
        item_embeddings = all_item_embeddings[item_indices]  # (batch_size, embedding_dim)
        
        # Aggregate members to group for each item in batch
        batch_size = item_embeddings.size(0)
        group_embeddings_list = []
        attention_weights_list = []
        
        for i in range(batch_size):
            # Get group embedding with attention for this item
            group_emb, attn_weights = self.group_aggregator(
                member_embeddings, item_embeddings[i]
            )
            group_embeddings_list.append(group_emb)
            attention_weights_list.append(attn_weights)
        
        # Stack results
        group_embeddings = torch.stack(group_embeddings_list)  # (batch_size, embedding_dim)
        attention_weights = torch.stack(attention_weights_list)  # (batch_size, num_members)
        
        return group_embeddings, attention_weights
    
    def predict(self, user_embeddings, item_embeddings):
        """
        Predict interaction scores between users and items.
        
        Simple inner product for now (no loss function yet).
        
        Args:
            user_embeddings: User representations, shape (batch_size, embedding_dim)
            item_embeddings: Item representations, shape (batch_size, embedding_dim)
        
        Returns:
            scores: Predicted interaction scores, shape (batch_size,)
        """
        # Inner product: score = u^T * i
        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return scores
    
    def get_all_embeddings(self, user_item_edges, kg_edges):
        """
        Get knowledge-aware embeddings for all users and entities after CKG propagation.
        
        Useful for inference and evaluation on the full dataset.
        
        Args:
            user_item_edges: All user-item interaction edges, shape (2, num_ui_edges)
            kg_edges: All knowledge graph edges, shape (2, num_kg_edges)
        
        Returns:
            all_user_embeddings: Knowledge-aware embeddings for all users
            all_entity_embeddings: Knowledge-aware embeddings for all entities
        """
        # Get all embeddings by doing a full forward pass
        all_users = torch.arange(self.num_users, dtype=torch.long)
        all_entities = torch.arange(self.num_entities, dtype=torch.long)
        
        user_embeds, entity_embeds = self.forward(
            user_indices=all_users,
            item_indices=all_entities,
            user_item_edges=user_item_edges,
            kg_edges=kg_edges
        )
        
        return user_embeds, entity_embeds
