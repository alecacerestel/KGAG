"""
KGAG: Knowledge-Aware Group Representation Learning for Group Recommendation

This package implements the KGAG model for group recommendation using knowledge graphs.

Main Components:
- model.py: Main KGAG model implementation
- layers.py: GCN layers for knowledge graph propagation
- aggregators.py: Group aggregation mechanisms (attention-based and baselines)
- losses.py: Loss functions including margin ranking loss

Usage Example:
    from KGAG import KGAG, AttentionGroupAggregator, MarginRankingLoss
    
    # Initialize model
    model = KGAG(
        num_users=1000,
        num_entities=5000,
        num_relations=10,
        embedding_dim=64,
        num_layers=3
    )
    
    # Initialize aggregator and loss
    aggregator = AttentionGroupAggregator(embedding_dim=64)
    loss_fn = MarginRankingLoss(margin=1.0)
"""

from .model import KGAG
from .layers import GCNLayer
from .aggregators import (
    AttentionGroupAggregator,
    SimpleGroupAggregator,
    HierarchicalGroupAggregator
)
from .losses import (
    MarginRankingLoss,
    BPRLoss,
    AdaptiveMarginLoss,
    ContrastiveLoss,
    CombinedLoss,
    TransELoss,
    compute_l2_reg,
    negative_sampling_loss
)

__all__ = [
    # Model
    'KGAG',
    
    # Layers
    'GCNLayer',
    
    # Aggregators
    'AttentionGroupAggregator',
    'SimpleGroupAggregator',
    'HierarchicalGroupAggregator',
    
    # Loss Functions
    'MarginRankingLoss',
    'BPRLoss',
    'AdaptiveMarginLoss',
    'ContrastiveLoss',
    'CombinedLoss',
    'TransELoss',
    
    # Utilities
    'compute_l2_reg',
    'negative_sampling_loss'
]

__version__ = '0.1.0'
