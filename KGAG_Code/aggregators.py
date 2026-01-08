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
    Attention-based Group Aggregator
    
    This aggregator learns member influence weights based on:
    1. Member-member similarity (learned from knowledge graph connectivity)
    2. Member relevance to candidate item (item-specific influence)
    
    The final group embedding is a weighted sum of member embeddings where
    weights are dynamically computed via attention mechanism.
    """
    
    def __init__(self, embedding_dim, attention_type='concat', dropout=0.1):
        """
        Initialize Attention-based Group Aggregator
        
        Args:
            embedding_dim (int): Dimension of user/item embeddings
            attention_type (str): Type of attention mechanism
                - 'concat': Concatenation-based attention (default)
                - 'dot': Dot-product attention
                - 'general': General bilinear attention
            dropout (float): Dropout probability for regularization
        """
        super(AttentionGroupAggregator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_type = attention_type
        
        # Attention networks based on type
        if attention_type == 'concat':
            # Concatenation-based attention: a = W * [u_i || u_j || v]
            # Input: [member_i, member_j, item] concatenated
            # Output: scalar attention score
            self.attention_layer = nn.Sequential(
                nn.Linear(3 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 1)
            )
            
        elif attention_type == 'dot':
            # Dot-product attention: a = (u_i * W_q)^T * (v * W_k)
            self.query_transform = nn.Linear(embedding_dim, embedding_dim)
            self.key_transform = nn.Linear(embedding_dim, embedding_dim)
            
        elif attention_type == 'general':
            # General bilinear attention: a = u_i^T * W * v
            self.attention_weight = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def compute_pairwise_similarity(self, member_embeddings):
        """
        Compute pairwise similarity between group members
        
        This captures user-user interaction and interest similarity based on
        their knowledge-aware embeddings learned from CKG propagation.
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
                              Embeddings of users in the group
        
        Returns:
            similarity_matrix: Tensor of shape (num_members, num_members)
                              Cosine similarity between each pair of members
        """
        # Normalize embeddings for cosine similarity
        norm_embeddings = F.normalize(member_embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarity: S = norm_embeddings * norm_embeddings^T
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        return similarity_matrix
    
    def compute_attention_weights_concat(self, member_embeddings, item_embedding, 
                                        pairwise_similarity=None):
        """
        Compute attention weights using concatenation-based attention
        
        For each member u_i in the group, compute attention score based on:
        1. The member's own embedding u_i
        2. Average embedding of other members (group context)
        3. The candidate item embedding v
        4. Optional: pairwise similarity with other members
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
            item_embedding: Tensor of shape (embedding_dim,)
            pairwise_similarity: Optional similarity matrix (num_members, num_members)
        
        Returns:
            attention_weights: Tensor of shape (num_members,)
                              Normalized attention weights (sum to 1)
        """
        num_members = member_embeddings.size(0)
        
        # Expand item embedding to match batch dimension
        item_expanded = item_embedding.unsqueeze(0).expand(num_members, -1)
        
        # For each member, compute their context (average of other members)
        # This captures the group composition influence
        member_sum = member_embeddings.sum(dim=0, keepdim=True)  # (1, embedding_dim)
        
        # For member i, context = (sum of all members - member i) / (num_members - 1)
        context_embeddings = (member_sum - member_embeddings) / max(num_members - 1, 1)
        
        # Concatenate: [member_i, context, item]
        # Shape: (num_members, 3 * embedding_dim)
        concat_input = torch.cat([
            member_embeddings,      # Member's own embedding
            context_embeddings,     # Other members' context
            item_expanded          # Candidate item
        ], dim=1)
        
        # Compute attention scores through MLP
        attention_scores = self.attention_layer(concat_input).squeeze(1)  # (num_members,)
        
        # Optional: Incorporate pairwise similarity as a bias term
        if pairwise_similarity is not None:
            # Average similarity of member i with all other members
            similarity_scores = pairwise_similarity.sum(dim=1) / num_members
            attention_scores = attention_scores + similarity_scores
        
        # Normalize attention scores to weights (softmax)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        return attention_weights
    
    def compute_attention_weights_dot(self, member_embeddings, item_embedding):
        """
        Compute attention weights using dot-product attention
        
        Inspired by Transformer-style attention:
        - Query: Transformed member embeddings
        - Key: Transformed item embedding
        - Attention: softmax(Q * K^T / sqrt(d))
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
            item_embedding: Tensor of shape (embedding_dim,)
        
        Returns:
            attention_weights: Tensor of shape (num_members,)
        """
        # Transform member embeddings to query space
        query = self.query_transform(member_embeddings)  # (num_members, embedding_dim)
        
        # Transform item embedding to key space
        key = self.key_transform(item_embedding)  # (embedding_dim,)
        
        # Compute dot-product attention scores
        # scores = Q * K^T
        attention_scores = torch.mv(query, key)  # (num_members,)
        
        # Scale by sqrt(embedding_dim) for stability
        attention_scores = attention_scores / (self.embedding_dim ** 0.5)
        
        # Normalize to weights
        attention_weights = F.softmax(attention_scores, dim=0)
        
        return attention_weights
    
    def compute_attention_weights_general(self, member_embeddings, item_embedding):
        """
        Compute attention weights using general bilinear attention
        
        Attention score: a_i = u_i^T * W * v
        where W is a learned weight matrix
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
            item_embedding: Tensor of shape (embedding_dim,)
        
        Returns:
            attention_weights: Tensor of shape (num_members,)
        """
        # Transform item embedding: W * v
        transformed_item = self.attention_weight(item_embedding)  # (embedding_dim,)
        
        # Compute bilinear scores: u_i^T * (W * v)
        attention_scores = torch.mv(member_embeddings, transformed_item)  # (num_members,)
        
        # Normalize to weights
        attention_weights = F.softmax(attention_scores, dim=0)
        
        return attention_weights
    
    def forward(self, member_embeddings, item_embedding, compute_similarity=True):
        """
        Aggregate member embeddings into group representation using attention
        
        Paper Foundation:
        "We propose an attention mechanism that can learn the influence of each user
         in a group to better model the group decision-making process. We take both
         user-user interaction and the influence of candidate item in group into
         consideration"
        
        Args:
            member_embeddings: Tensor of shape (num_members, embedding_dim)
                              Knowledge-aware embeddings of group members
            item_embedding: Tensor of shape (embedding_dim,)
                           Knowledge-aware embedding of candidate item
            compute_similarity: Whether to compute pairwise member similarity
        
        Returns:
            group_embedding: Tensor of shape (embedding_dim,)
                            Weighted aggregation of member embeddings
            attention_weights: Tensor of shape (num_members,)
                              Attention weight for each member (for interpretability)
        """
        # Compute pairwise similarity if requested
        pairwise_similarity = None
        if compute_similarity and self.attention_type == 'concat':
            pairwise_similarity = self.compute_pairwise_similarity(member_embeddings)
        
        # Compute attention weights based on selected mechanism
        if self.attention_type == 'concat':
            attention_weights = self.compute_attention_weights_concat(
                member_embeddings, item_embedding, pairwise_similarity
            )
        elif self.attention_type == 'dot':
            attention_weights = self.compute_attention_weights_dot(
                member_embeddings, item_embedding
            )
        elif self.attention_type == 'general':
            attention_weights = self.compute_attention_weights_general(
                member_embeddings, item_embedding
            )
        
        # Apply dropout to attention weights for regularization
        attention_weights = self.dropout(attention_weights)
        
        # Renormalize after dropout
        attention_weights = attention_weights / (attention_weights.sum() + 1e-10)
        
        # Aggregate member embeddings with attention weights
        # group_embedding = sum_i (attention_weight_i * member_embedding_i)
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
    
    def __init__(self, embedding_dim, num_subgroups=2, attention_type='concat'):
        """
        Initialize hierarchical aggregator
        
        Args:
            embedding_dim (int): Dimension of embeddings
            num_subgroups (int): Number of subgroups to form
            attention_type (str): Type of attention for aggregation
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
        
        # Attention aggregator for within-subgroup aggregation
        self.subgroup_aggregator = AttentionGroupAggregator(
            embedding_dim, attention_type
        )
        
        # Attention aggregator for cross-subgroup aggregation
        self.group_aggregator = AttentionGroupAggregator(
            embedding_dim, attention_type
        )
    
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
