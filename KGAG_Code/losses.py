"""
Loss Functions for Knowledge-Aware Group Recommendation (KGAG)

This module implements various loss functions for training the KGAG model,
with emphasis on the margin-based ranking loss proposed in the paper.

Paper Foundation (Section I, Contribution 3):
"To learn more discriminative representations of groups and items, we also extend
 the idea of margin loss which not only requires the score of the positive sample
 to be higher than that of the negative sample, but also requires the score of
 the positive sample is higher than that of the negative sample by a given margin"

Key Insight:
Traditional BPR loss: score(positive) > score(negative)
Margin loss: score(positive) > score(negative) + margin
This enforces a stricter separation, leading to more discriminative embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """
    Margin-based Ranking Loss for Group Recommendation
    
    Paper Foundation (Section I, Contribution 3):
    "we also extend the idea of margin loss which not only requires the score of
     the positive sample to be higher than that of the negative sample, but also
     requires the score of the positive sample is higher than that of the negative
     sample by a given margin"
    
    Mathematical Formulation:
        L_margin = max(0, margin - (score_pos - score_neg))
        
    Or equivalently:
        L_margin = max(0, margin + score_neg - score_pos)
    
    This ensures: score_pos >= score_neg + margin
    
    Benefits:
    1. Stronger separation between positive and negative samples
    2. More discriminative embeddings
    3. Better generalization to unseen groups/items
    """
    
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Initialize Margin Ranking Loss
        
        Args:
            margin (float): Minimum required difference between positive and negative scores
                           Typical values: 0.5, 1.0, 2.0
                           Larger margin -> stricter separation
            reduction (str): How to reduce batch losses
                - 'mean': Average loss over batch (default)
                - 'sum': Sum of losses
                - 'none': Return individual losses
        """
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, pos_scores, neg_scores):
        """
        Compute margin ranking loss
        
        Args:
            pos_scores: Scores for positive (group, item) pairs
                       Shape: (batch_size,)
            neg_scores: Scores for negative (group, item) pairs
                       Shape: (batch_size,) or (batch_size, num_negatives)
        
        Returns:
            loss: Margin ranking loss (scalar if reduction='mean' or 'sum')
        """
        # Handle multiple negatives per positive
        if neg_scores.dim() == 2:
            # Multiple negative samples: (batch_size, num_negatives)
            # Expand pos_scores to match: (batch_size, 1)
            pos_scores = pos_scores.unsqueeze(1)
        
        # Compute margin violation: max(0, margin + neg_score - pos_score)
        # If pos_score > neg_score + margin, no violation (loss = 0)
        # Otherwise, penalize the violation
        losses = torch.clamp(self.margin + neg_scores - pos_scores, min=0.0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) Loss
    
    Classic pairwise ranking loss for implicit feedback.
    Serves as a baseline to compare against margin loss.
    
    Mathematical Formulation:
        L_BPR = -log(sigmoid(score_pos - score_neg))
        
    Or equivalently:
        L_BPR = log(1 + exp(-(score_pos - score_neg)))
        
    This is equivalent to:
        L_BPR = softplus(score_neg - score_pos)
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize BPR Loss
        
        Args:
            reduction (str): How to reduce batch losses
        """
        super(BPRLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pos_scores, neg_scores):
        """
        Compute BPR loss
        
        Args:
            pos_scores: Scores for positive pairs, shape (batch_size,)
            neg_scores: Scores for negative pairs, shape (batch_size,) or (batch_size, num_negatives)
        
        Returns:
            loss: BPR loss
        """
        # Handle multiple negatives
        if neg_scores.dim() == 2:
            pos_scores = pos_scores.unsqueeze(1)
        
        # BPR loss: -log(sigmoid(pos - neg)) = softplus(neg - pos)
        losses = F.softplus(neg_scores - pos_scores)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class AdaptiveMarginLoss(nn.Module):
    """
    Adaptive Margin Loss (Advanced variant)
    
    Dynamically adjusts margin based on:
    1. Score confidence (higher confidence -> larger margin)
    2. Training progress (curriculum learning)
    
    This helps with:
    - Easy samples get larger margins (push further apart)
    - Hard samples get smaller margins (avoid overfitting)
    """
    
    def __init__(self, base_margin=1.0, adaptive_factor=0.5, reduction='mean'):
        """
        Initialize Adaptive Margin Loss
        
        Args:
            base_margin (float): Base margin value
            adaptive_factor (float): How much to adapt margin (0-1)
                                    0 = no adaptation (fixed margin)
                                    1 = full adaptation
            reduction (str): Reduction method
        """
        super(AdaptiveMarginLoss, self).__init__()
        self.base_margin = base_margin
        self.adaptive_factor = adaptive_factor
        self.reduction = reduction
    
    def forward(self, pos_scores, neg_scores):
        """
        Compute adaptive margin loss
        
        Margin is adapted based on score confidence:
        - If pos_score is very high, increase margin (push harder)
        - If pos_score is moderate, use base margin
        
        Args:
            pos_scores: Positive scores, shape (batch_size,)
            neg_scores: Negative scores, shape (batch_size,) or (batch_size, num_negatives)
        
        Returns:
            loss: Adaptive margin loss
        """
        # Handle multiple negatives
        if neg_scores.dim() == 2:
            pos_scores_expanded = pos_scores.unsqueeze(1)
        else:
            pos_scores_expanded = pos_scores
        
        # Compute adaptive margin based on positive score confidence
        # Higher pos_score -> higher margin (push further)
        # Use sigmoid to bound the adaptation
        confidence = torch.sigmoid(pos_scores)
        adaptive_margin = self.base_margin * (1.0 + self.adaptive_factor * confidence.unsqueeze(1))
        
        # Compute margin loss with adaptive margin
        losses = torch.clamp(adaptive_margin + neg_scores - pos_scores_expanded, min=0.0)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Group-Item Pairs
    
    Treats the problem as a binary classification:
    - Positive pairs should have high scores (close to 1)
    - Negative pairs should have low scores (close to 0)
    
    Uses binary cross-entropy loss.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize Contrastive Loss
        
        Args:
            reduction (str): Reduction method
        """
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pos_scores, neg_scores):
        """
        Compute contrastive loss
        
        Args:
            pos_scores: Positive scores, shape (batch_size,)
            neg_scores: Negative scores, shape (batch_size,) or (batch_size, num_negatives)
        
        Returns:
            loss: Binary cross-entropy loss
        """
        # Apply sigmoid to convert scores to probabilities
        pos_probs = torch.sigmoid(pos_scores)
        neg_probs = torch.sigmoid(neg_scores)
        
        # Positive loss: -log(pos_prob)
        pos_loss = -torch.log(pos_probs + 1e-10)
        
        # Negative loss: -log(1 - neg_prob)
        neg_loss = -torch.log(1 - neg_probs + 1e-10)
        
        # Combine losses
        total_loss = pos_loss.mean() + neg_loss.mean()
        
        if self.reduction == 'mean':
            return total_loss
        elif self.reduction == 'sum':
            return total_loss * pos_scores.size(0)
        else:
            return torch.cat([pos_loss, neg_loss.flatten()])


class CombinedLoss(nn.Module):
    """
    Combined Loss Function
    
    Combines multiple loss components:
    1. Margin/BPR loss for ranking
    2. Regularization loss (L2 penalty on embeddings)
    3. Optional: Knowledge graph embedding loss (for entity/relation embeddings)
    
    Total Loss:
        L_total = L_ranking + λ_reg * L_reg + λ_kg * L_kg
    """
    
    def __init__(self, ranking_loss='margin', margin=1.0, 
                 reg_weight=1e-5, kg_weight=0.0):
        """
        Initialize Combined Loss
        
        Args:
            ranking_loss (str): Type of ranking loss ('margin', 'bpr', 'adaptive', 'contrastive')
            margin (float): Margin value for margin-based losses
            reg_weight (float): Weight for L2 regularization
            kg_weight (float): Weight for knowledge graph embedding loss
        """
        super(CombinedLoss, self).__init__()
        
        # Select ranking loss
        if ranking_loss == 'margin':
            self.ranking_loss = MarginRankingLoss(margin=margin)
        elif ranking_loss == 'bpr':
            self.ranking_loss = BPRLoss()
        elif ranking_loss == 'adaptive':
            self.ranking_loss = AdaptiveMarginLoss(base_margin=margin)
        elif ranking_loss == 'contrastive':
            self.ranking_loss = ContrastiveLoss()
        else:
            raise ValueError(f"Unknown ranking loss: {ranking_loss}")
        
        self.reg_weight = reg_weight
        self.kg_weight = kg_weight
    
    def forward(self, pos_scores, neg_scores, embeddings=None, kg_loss=None):
        """
        Compute combined loss
        
        Args:
            pos_scores: Positive scores, shape (batch_size,)
            neg_scores: Negative scores, shape (batch_size,) or (batch_size, num_negatives)
            embeddings: List of embeddings to regularize (optional)
                       Example: [user_embeddings, item_embeddings, group_embeddings]
            kg_loss: Knowledge graph embedding loss (optional)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Compute ranking loss
        ranking_loss_value = self.ranking_loss(pos_scores, neg_scores)
        
        # Compute regularization loss (L2 penalty on embeddings)
        reg_loss_value = 0.0
        if embeddings is not None and self.reg_weight > 0:
            for emb in embeddings:
                # L2 regularization: ||embedding||^2
                reg_loss_value += torch.norm(emb, p=2) ** 2
            reg_loss_value = reg_loss_value / len(embeddings)
        
        # Compute knowledge graph loss
        kg_loss_value = 0.0
        if kg_loss is not None and self.kg_weight > 0:
            kg_loss_value = kg_loss
        
        # Combine all losses
        total_loss = (
            ranking_loss_value + 
            self.reg_weight * reg_loss_value + 
            self.kg_weight * kg_loss_value
        )
        
        # Return total loss and individual components for logging
        loss_dict = {
            'total': total_loss.item(),
            'ranking': ranking_loss_value.item(),
            'regularization': reg_loss_value if isinstance(reg_loss_value, float) else reg_loss_value.item(),
            'kg': kg_loss_value if isinstance(kg_loss_value, float) else kg_loss_value.item()
        }
        
        return total_loss, loss_dict


class TransELoss(nn.Module):
    """
    TransE Loss for Knowledge Graph Embeddings
    
    Used to learn better entity and relation embeddings from KG triples.
    Can be used as an auxiliary loss component in the combined loss.
    
    For a KG triple (head, relation, tail):
        score = ||head + relation - tail||
        
    Positive triples should have low score (close entities)
    Negative triples should have high score (distant entities)
    
    Loss:
        L_TransE = max(0, margin + score_pos - score_neg)
    """
    
    def __init__(self, margin=1.0, norm='L2', reduction='mean'):
        """
        Initialize TransE Loss
        
        Args:
            margin (float): Margin for separating positive and negative triples
            norm (str): Distance norm ('L1' or 'L2')
            reduction (str): Reduction method
        """
        super(TransELoss, self).__init__()
        self.margin = margin
        self.norm = norm
        self.reduction = reduction
    
    def compute_score(self, head, relation, tail):
        """
        Compute TransE score: ||head + relation - tail||
        
        Args:
            head: Head entity embeddings, shape (batch_size, embedding_dim)
            relation: Relation embeddings, shape (batch_size, embedding_dim)
            tail: Tail entity embeddings, shape (batch_size, embedding_dim)
        
        Returns:
            score: Distance score, shape (batch_size,)
        """
        # TransE translation: head + relation ≈ tail
        # Score = distance between (head + relation) and tail
        translated = head + relation - tail
        
        if self.norm == 'L1':
            score = torch.norm(translated, p=1, dim=1)
        else:  # L2
            score = torch.norm(translated, p=2, dim=1)
        
        return score
    
    def forward(self, pos_head, pos_relation, pos_tail, 
                neg_head, neg_relation, neg_tail):
        """
        Compute TransE loss
        
        Args:
            pos_head, pos_relation, pos_tail: Positive triple embeddings
            neg_head, neg_relation, neg_tail: Negative triple embeddings
            All shapes: (batch_size, embedding_dim)
        
        Returns:
            loss: TransE margin loss
        """
        # Compute scores for positive and negative triples
        pos_score = self.compute_score(pos_head, pos_relation, pos_tail)
        neg_score = self.compute_score(neg_head, neg_relation, neg_tail)
        
        # Margin loss: encourage pos_score < neg_score - margin
        losses = torch.clamp(self.margin + pos_score - neg_score, min=0.0)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


# Utility functions for loss computation

def compute_l2_reg(embeddings):
    """
    Compute L2 regularization loss for embeddings
    
    Args:
        embeddings: Single embedding tensor or list of embedding tensors
    
    Returns:
        reg_loss: L2 regularization loss
    """
    if isinstance(embeddings, (list, tuple)):
        reg_loss = sum(torch.norm(emb, p=2) ** 2 for emb in embeddings) / len(embeddings)
    else:
        reg_loss = torch.norm(embeddings, p=2) ** 2
    
    return reg_loss


def negative_sampling_loss(model, users, pos_items, neg_items, 
                           loss_fn, num_negatives=1):
    """
    Compute loss with negative sampling
    
    For each (user, positive_item) pair, sample negative items and compute loss.
    
    Args:
        model: KGAG model
        users: User indices, shape (batch_size,)
        pos_items: Positive item indices, shape (batch_size,)
        neg_items: Negative item indices, shape (batch_size, num_negatives)
        loss_fn: Loss function (e.g., MarginRankingLoss)
        num_negatives: Number of negative samples per positive
    
    Returns:
        loss: Computed loss value
    """
    batch_size = users.size(0)
    
    # Get user and item embeddings
    user_embeddings = model.user_embedding(users)  # (batch_size, embedding_dim)
    pos_item_embeddings = model.entity_embedding(pos_items)  # (batch_size, embedding_dim)
    
    # Compute positive scores
    pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)  # (batch_size,)
    
    # Compute negative scores
    if num_negatives > 1:
        # Multiple negatives: (batch_size, num_negatives)
        neg_item_embeddings = model.entity_embedding(neg_items)  # (batch_size, num_negatives, embedding_dim)
        neg_scores = (user_embeddings.unsqueeze(1) * neg_item_embeddings).sum(dim=2)
    else:
        # Single negative: (batch_size,)
        neg_item_embeddings = model.entity_embedding(neg_items.squeeze(1))
        neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)
    
    # Compute loss
    loss = loss_fn(pos_scores, neg_scores)
    
    return loss
