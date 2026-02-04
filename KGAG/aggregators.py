"""
Group Aggregation Module for Knowledge-Aware Group Recommendation (KGAG)

This module implements the attention-based group aggregation mechanism that combines
individual member preferences into a unified group representation.

Key Features (from paper):
1. User-User Interaction Consideration: Members with similar interests have stronger influence
2. Candidate Item Awareness: Member influence varies based on the specific item being recommended
3. Dynamic Attention: Influence weights are computed dynamically, not static social network weights

Paper Foundation (Section I):
"We propose an attention mechanism that can learn the influence of each user in a group
 to better model the group decision-making process. We take both user-user interaction
 and the influence of candidate item in group into consideration"

Key Insight (Section I, para 6):
"the influence of each member on the final result should be different and it varies
 according to other group members and candidate item"

Why not social networks? (Section I, para 6):
"a person who is active in the group may not know much about movies. When this group
 wants to see a movie, she will also consult other members of the group who know better
 about movies. Therefore, she does not necessarily play a major role in this group's
 decision making"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGroupAggregator(nn.Module):
    """
    Attention-based Group Aggregator implementing the paper's SP + PI mechanism.

    Paper Foundation (Section III-C: Attention Mechanism):
    The attention weight for each member is computed from two components:

    1. SP (Self Persistence): Member's own preference for the candidate item
       - Captures direct user-item relevance via inner product fpref(ui, v) = ui^T v

    2. PI (Peer Influence): Influence from other members in the same group
       - Peer set SPg,i = {u | u ∈ g, u ≠ ui}
       - αPI(g,i) = v_c^T Φ(W_c1 ui + W_c2 CONCAT_{u∈SPg,i}(u) + b)

    Final attention weight:
       a_i = softmax(α * αSP(g,i,v) + β * αPI(g,i))

    Group embedding:
       e_g = Σ_i a_i * e_ui
    """

    def __init__(self, embedding_dim, dropout=0.1):
        """Initialize Attention-based Group Aggregator (paper mechanism only).

        Args:
            embedding_dim (int): Dimension of user/item embeddings
            dropout (float): Dropout probability for regularization
        """
        super(AttentionGroupAggregator, self).__init__()

        self.embedding_dim = embedding_dim

        # Paper's mechanism: SP (self persistence) + PI (peer influence)

        # SP uses simple inner product: fpref(ui, v) = ui^T * v
        # No additional parameters needed

        # PI formula from paper: αPI(g,i) = vc^T * Φ(Wc1*ui + Wc2*CONCAT(peers) + b)
        # Wc1: transform user embedding
        self.W_c1 = nn.Linear(embedding_dim, embedding_dim)

        # Wc2: transform concatenated peer embeddings
        # Paper: Wc2 ∈ R^(d×d·|SPg,i|) but groups have variable sizes
        # Practical solution: pool peers via attention to fixed d-dim, then apply Wc2
        self.peer_attention = nn.Linear(embedding_dim, 1)  # Attention score per peer
        self.W_c2 = nn.Linear(embedding_dim, embedding_dim)

        # vc: final projection vector to scalar score
        self.v_c = nn.Linear(embedding_dim, 1)

        # Learnable weights for balancing SP and PI
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)  # Weight for SP
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=False)   # Weight for PI

        # Activation function Φ
        self.activation = nn.ReLU()

        # Dropout for regularization
        self.pi_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def compute_attention_weights_paper(self, member_embeddings, item_embedding):
        """
        Compute attention weights using paper's SP + PI mechanism
        
        Paper Formulas (Section III-C):
        
        1. Self Persistence (SP):
           αSP(g,i,v) = fpref(ui,v) = ui^T * v
           Inner product between user and item embeddings
        
        2. Peer Influence (PI):
           αPI(g,i) = vc^T * Φ(Wc1*ui + Wc2*CONCAT_{u∈SPg,i}(u) + b)
           Paper definition: "The peer set of ui in group g is a set which 
           includes all the members in g except user ui"
           SPg,i = {u | u ∈ g, u ≠ ui}
           Where:
           - SPg,i: ALL other members in group g (not similarity-based)
           - CONCAT: concatenate all peer embeddings
           - Wc1, Wc2: weight matrices
           - vc: projection vector
           - Φ: activation function (ReLU)
        
        3. Final attention:
           a_i = softmax(α * αSP(g,i,v) + β * αPI(g,i))
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
            item_embedding: Tensor of shape (embedding_dim,)
        
        Returns:
            attention_weights: Tensor of shape (num_members,)
                              Normalized attention weights (sum to 1)
        """
        num_members = member_embeddings.size(0)
        
        # ===================================================================
        # STEP 1: Compute SP (Self Persistence) using inner product
        # ===================================================================
        # αSP(g,i,v) = fpref(ui,v) = ui^T * v
        # Simple dot product between each user embedding and item embedding
        
        sp_scores = torch.mv(member_embeddings, item_embedding)  # (num_members,)
        
        # ===================================================================
        # STEP 2: Compute PI (Peer Influence) for each member
        # ===================================================================
        # αPI(g,i) = vc^T * Φ(Wc1*ui + Wc2*CONCAT_{u∈SPg,i}(u) + b)
        # SPg,i = all other members in group except ui
        
        # 2b. For each member, compute peer influence
        pi_scores = []
        
        for i in range(num_members):
            # Get user i's embedding
            ui = member_embeddings[i]  # (embedding_dim,)
            
            # Get peer set SPg,i: all other members in group (exclude self)
            mask = torch.ones(num_members, dtype=torch.bool, device=member_embeddings.device)
            mask[i] = False  # Exclude self
            
            peer_embeddings = member_embeddings[mask]  # (num_members-1, embedding_dim)
            
            # CONCAT all peer embeddings (variable size: |SPg,i| × d)
            # Since Wc2 expects fixed input and |SPg,i| varies by group size,
            # we use attention pooling to get a fixed d-dimensional representation
            # This approximates the paper's CONCAT operation while handling variable sizes
            
            if peer_embeddings.size(0) > 0:
                # Compute attention scores for each peer
                peer_scores = self.peer_attention(peer_embeddings).squeeze(1)  # (num_peers,)
                peer_weights = F.softmax(peer_scores, dim=0)  # (num_peers,)
                
                # Attention-weighted pooling: approximates CONCAT + Wc2 for variable sizes
                peer_agg = torch.sum(
                    peer_weights.unsqueeze(1) * peer_embeddings,
                    dim=0
                )  # (embedding_dim,)
            else:
                # Edge case: single-member group (no peers)
                peer_agg = torch.zeros_like(ui)
            
            # Apply paper's formula: vc^T * Φ(Wc1*ui + Wc2*peer_agg + b)
            transformed_ui = self.W_c1(ui)  # (embedding_dim,)
            transformed_peers = self.W_c2(peer_agg)  # (embedding_dim,)
            
            combined = transformed_ui + transformed_peers  # bias is in Linear layers
            activated = self.activation(combined)  # Φ (ReLU)
            activated = self.pi_dropout(activated)  # Dropout
            
            pi_score = self.v_c(activated).squeeze()  # scalar
            pi_scores.append(pi_score)
        
        pi_scores = torch.stack(pi_scores)  # (num_members,)
        
        # ===================================================================
        # STEP 3: Combine SP and PI with learnable weights
        # ===================================================================
        # Combined score: α * αSP + β * αPI
        combined_scores = self.alpha * sp_scores + self.beta * pi_scores
        
        # ===================================================================
        # STEP 4: Normalize to attention weights (softmax)
        # ===================================================================
        attention_weights = F.softmax(combined_scores, dim=0)
        
        return attention_weights
    def forward(self, member_embeddings, item_embedding):
        """
        Aggregate member embeddings into group representation using attention
        
        Paper Foundation (Section III-C: Attention Mechanism):
        "We propose an attention mechanism that can learn the influence of each user
         in a group to better model the group decision-making process. We take both
         user-user interaction and the influence of candidate item in group into
         consideration"
        
        Mechanism:
        1. Compute attention weights: a_i = softmax(α * SP_i + β * PI_i)
           - SP_i: Self persistence (member i's preference for item)
           - PI_i: Peer influence (weighted by member similarity)
        
        2. Aggregate: e_g = Σ_i a_i * e_ui (group preference)
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
                              Knowledge-aware embeddings of group members
            item_embedding: Tensor of shape (embedding_dim,)
                           Knowledge-aware embedding of candidate item
        
        Returns:
            group_embedding: Tensor of shape (embedding_dim,)
                            Weighted aggregation of member embeddings (e_g)
            attention_weights: Tensor of shape (num_members,)
                              Attention weight for each member (a_i, for interpretability)
        """
        # Compute attention weights using paper's SP + PI mechanism
        attention_weights = self.compute_attention_weights_paper(
            member_embeddings, item_embedding
        )
        
        # # Apply dropout to attention weights for regularization
        # attention_weights = self.dropout(attention_weights)
        
        # # Renormalize after dropout
        # attention_weights = attention_weights / (attention_weights.sum() + 1e-10)
        
        # ===================================================================
        # FINAL STEP: Compute group preference (e_g)
        # ===================================================================
        # e_g = Σ_i a_i * e_ui
        # Weighted sum of member embeddings using learned attention weights
        group_embedding = torch.sum(
            attention_weights.unsqueeze(1) * member_embeddings,
            dim=0
        )
        
        return group_embedding, attention_weights


class SimpleGroupAggregator(nn.Module):
    """
    Simple baseline aggregators for comparison
    
    These serve as baselines to demonstrate the effectiveness of attention mechanism:
    - Mean: Simple average of member embeddings
    - Max: Element-wise maximum of member embeddings
    - Min: Element-wise minimum of member embeddings
    """
    
    def __init__(self, aggregation_type='mean'):
        """
        Initialize simple aggregator
        
        Args:
            aggregation_type (str): Type of aggregation
                - 'mean': Average pooling
                - 'max': Max pooling
                - 'min': Min pooling
        """
        super(SimpleGroupAggregator, self).__init__()
        self.aggregation_type = aggregation_type
    
    def forward(self, member_embeddings, item_embedding=None):
        """
        Aggregate member embeddings using simple pooling
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
            item_embedding: Not used (for API consistency with attention aggregator)
        
        Returns:
            group_embedding: Tensor of shape (embedding_dim,)
            None: No attention weights for simple aggregators
        """
        if self.aggregation_type == 'mean':
            group_embedding = torch.mean(member_embeddings, dim=0)
        elif self.aggregation_type == 'max':
            group_embedding, _ = torch.max(member_embeddings, dim=0)
        elif self.aggregation_type == 'min':
            group_embedding, _ = torch.min(member_embeddings, dim=0)
        
        return group_embedding, None


class HierarchicalGroupAggregator(nn.Module):
    """
    Hierarchical Group Aggregator (Advanced variant)
    
    This aggregator first clusters members into subgroups based on similarity,
    then aggregates within subgroups before final group-level aggregation.
    
    Useful for large groups with diverse preferences.
    """
    
    def __init__(self, embedding_dim, num_subgroups=2):
        """Initialize hierarchical aggregator.

        Args:
            embedding_dim (int): Dimension of embeddings
            num_subgroups (int): Number of subgroups to form
        """
        super(HierarchicalGroupAggregator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_subgroups = num_subgroups
        
        # Subgroup assignment network
        self.subgroup_assignment = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_subgroups),
            nn.Softmax(dim=1)
        )
        
        # Attention aggregators (paper SP + PI) for intra- and inter-subgroup aggregation
        self.subgroup_aggregator = AttentionGroupAggregator(embedding_dim)
        self.group_aggregator = AttentionGroupAggregator(embedding_dim)
    
    def forward(self, member_embeddings, item_embedding):
        """
        Hierarchical aggregation: subgroups -> group
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
            item_embedding: Tensor of shape (embedding_dim,)
        
        Returns:
            group_embedding: Tensor of shape (embedding_dim,)
            subgroup_info: Dict containing subgroup assignments and attention weights
        """
        num_members = member_embeddings.size(0)
        
        # Step 1: Assign members to subgroups (soft assignment)
        subgroup_weights = self.subgroup_assignment(member_embeddings)  # (num_members, num_subgroups)
        
        # Step 2: Aggregate within each subgroup
        subgroup_embeddings = []
        for subgroup_idx in range(self.num_subgroups):
            # Get soft membership weights for this subgroup
            weights = subgroup_weights[:, subgroup_idx].unsqueeze(1)  # (num_members, 1)
            
            # Weighted member embeddings for this subgroup
            weighted_members = weights * member_embeddings
            
            # Aggregate using attention (if more than one member has non-zero weight)
            if weights.sum() > 1e-5:
                subgroup_emb, _ = self.subgroup_aggregator(
                    weighted_members, item_embedding
                )
            else:
                # Fallback to mean if subgroup is empty
                subgroup_emb = weighted_members.mean(dim=0)
            
            subgroup_embeddings.append(subgroup_emb)
        
        # Step 3: Aggregate subgroup embeddings to final group embedding
        subgroup_embeddings = torch.stack(subgroup_embeddings)  # (num_subgroups, embedding_dim)
        
        group_embedding, subgroup_attention = self.group_aggregator(
            subgroup_embeddings, item_embedding
        )
        
        # Return results with subgroup information for interpretability
        subgroup_info = {
            'subgroup_assignments': subgroup_weights,
            'subgroup_attention': subgroup_attention
        }
        
        return group_embedding, subgroup_info
