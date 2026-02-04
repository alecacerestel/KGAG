# KGAG: Knowledge-Aware Group Representation Learning for Group Recommendation

This directory contains the implementation of KGAG (Knowledge-Aware Group Recommendation with Attention-based Group aggregation), a deep learning model for group recommendation that leverages knowledge graphs.

## 📁 File Structure

```
KGAG/
├── __init__.py              # Package initialization and exports
├── model.py                 # Main KGAG model
├── layers.py                # GCN layers for knowledge graph propagation
├── aggregators.py           # Group aggregation mechanisms (NEW)
├── losses.py                # Loss functions including margin ranking loss (NEW)
├── example_usage.py         # Usage examples and tutorials (NEW)
└── README.md                # This file (NEW)
```

## 🎯 Key Features

### 1. Knowledge Graph Integration
- Constructs a **Collaborative Knowledge Graph (CKG)** merging user-item interactions with semantic knowledge graph
- Propagates information through GCN layers to capture high-order connectivity
- Learns knowledge-aware representations for users and items

### 2. Attention-based Group Aggregation
**Multiple aggregation strategies** (in `aggregators.py`):
- **AttentionGroupAggregator**: Dynamic attention mechanism considering:
  - User-user similarity (learned from knowledge graph connectivity)
  - Candidate item influence (item-specific member weights)
    - Paper mechanism: SP (Self Persistence) + PI (Peer Influence)
- **SimpleGroupAggregator**: Baselines (mean, max, min pooling)
- **HierarchicalGroupAggregator**: Multi-level aggregation for large groups

### 3. Margin-based Loss Functions
**Comprehensive loss functions** (in `losses.py`):
- **MarginRankingLoss**: Enforces strict separation between positive/negative samples
- **BPRLoss**: Classic Bayesian Personalized Ranking
- **AdaptiveMarginLoss**: Dynamic margin adjustment
- **ContrastiveLoss**: Binary classification approach
- **CombinedLoss**: Multi-component loss with regularization
- **TransELoss**: Knowledge graph embedding loss

## 🚀 Quick Start

### Basic Usage

```python
from KGAG import KGAG, AttentionGroupAggregator, MarginRankingLoss

# Initialize model
model = KGAG(
    num_users=1000,
    num_entities=5000,
    num_relations=10,
    embedding_dim=64,
    num_layers=3
)

# Prepare data (user-item edges and KG edges)
user_item_edges = torch.stack([user_ids, item_ids])  # (2, num_edges)
kg_edges = torch.stack([head_entities, tail_entities])  # (2, num_kg_edges)

# Forward pass
user_embeddings, item_embeddings = model(
    user_indices=batch_users,
    item_indices=batch_items,
    user_item_edges=user_item_edges,
    kg_edges=kg_edges
)

# Get group embeddings with attention
group_members = [1, 5, 12, 23]  # User IDs in the group
candidate_items = torch.tensor([10, 25, 50])

group_embeddings, attention_weights = model.get_group_embedding(
    group_members=group_members,
    item_indices=candidate_items,
    user_item_edges=user_item_edges,
    kg_edges=kg_edges
)

# Compute group-item scores
scores = (group_embeddings * item_embeddings).sum(dim=1)
```

### Training with Margin Loss

```python
from KGAG import CombinedLoss
import torch.optim as optim

# Initialize loss function
loss_fn = CombinedLoss(
    ranking_loss='margin',
    margin=1.0,
    reg_weight=1e-5
)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Forward pass (positive samples)
    user_emb_pos, item_emb_pos = model(users, pos_items, ui_edges, kg_edges)
    
    # Forward pass (negative samples)
    user_emb_neg, item_emb_neg = model(users, neg_items, ui_edges, kg_edges)
    
    # Compute scores
    pos_scores = model.predict(user_emb_pos, item_emb_pos)
    neg_scores = model.predict(user_emb_neg, item_emb_neg)
    
    # Compute loss
    loss, loss_dict = loss_fn(
        pos_scores, neg_scores,
        embeddings=[user_emb_pos, item_emb_pos]
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 📚 Components Overview

### Model Architecture (`model.py`)

**KGAG Class**: Main model implementing:
- **Phase 1**: Embedding initialization (users, entities, relations)
- **Phase 2**: Collaborative Knowledge Graph construction and GCN propagation
- **Phase 3**: Group aggregation with attention mechanism
- **Phase 4**: Prediction and optimization

Key methods:
- `forward()`: Propagate through CKG, get knowledge-aware embeddings
- `get_group_embedding()`: Aggregate members with attention
- `predict()`: Compute interaction scores
- `get_all_embeddings()`: Full propagation for inference

### GCN Layers (`layers.py`)

**GCNLayer Class**: Graph Convolution for knowledge propagation
- Supports multiple aggregation types: bi-interaction, GCN, GraphSAGE
- Implements neighbor aggregation and representation update
- Handles relation-aware message passing

### Group Aggregators (`aggregators.py`)

#### AttentionGroupAggregator
Paper-aligned attention mechanism that learns member influence based on:
- **Self Persistence (SP)**: Each member's own preference for the candidate item
    via inner product between user and item embeddings.
- **Peer Influence (PI)**: Influence from other members in the group, modeled as
    a neural transformation of the target member and an attention-pooled summary
    of their peers.

Final attention weight combines these components:
\( a_i = \mathrm{softmax}(\alpha \cdot \alpha_{SP}(g,i,v) + \beta \cdot \alpha_{PI}(g,i)) \),
and the group embedding is \( e_g = \sum_i a_i e_{u_i} \).

#### SimpleGroupAggregator
Baseline aggregators for comparison:
- **Mean**: Average pooling
- **Max**: Element-wise maximum
- **Min**: Element-wise minimum

#### HierarchicalGroupAggregator
Advanced multi-level aggregation:
1. Assign members to subgroups (soft clustering)
2. Aggregate within subgroups
3. Aggregate subgroups to final group embedding

### Loss Functions (`losses.py`)

#### MarginRankingLoss
**Core contribution from paper**:
```
L = max(0, margin + score_neg - score_pos)
```
Enforces: `score_pos >= score_neg + margin`

Benefits:
- Stronger separation between positive/negative samples
- More discriminative embeddings
- Better generalization

#### BPRLoss
Classic pairwise ranking:
```
L = -log(sigmoid(score_pos - score_neg))
```
Baseline for comparison.

#### AdaptiveMarginLoss
Dynamic margin based on score confidence:
- High confidence → larger margin (push harder)
- Low confidence → smaller margin (avoid overfitting)

#### CombinedLoss
Multi-component loss:
```
L_total = L_ranking + λ_reg * L_reg + λ_kg * L_kg
```
Components:
- Ranking loss (margin/BPR/adaptive/contrastive)
- L2 regularization on embeddings
- Optional KG embedding loss

#### TransELoss
Knowledge graph embedding loss:
```
L = max(0, margin + ||h + r - t|| - ||h' + r - t'||)
```
For learning better entity/relation embeddings.

## 🔧 Advanced Usage

### Group Attention Usage

```python
from KGAG.aggregators import AttentionGroupAggregator

# Create aggregator using the paper's SP + PI mechanism
aggregator = AttentionGroupAggregator(
    embedding_dim=64,
    dropout=0.1
)

# Get group embedding with attention weights
group_emb, attn_weights = aggregator(member_embeddings, item_embedding)

# Interpret attention weights
for i, member_id in enumerate(group_members):
    print(f"Member {member_id}: weight = {attn_weights[i]:.4f}")
```

### Multiple Negative Samples

```python
# Sample multiple negatives per positive
num_negatives = 5
pos_scores = model.predict(user_emb, pos_item_emb)  # (batch_size,)
neg_scores = model.predict(
    user_emb.unsqueeze(1).expand(-1, num_negatives, -1),
    neg_item_emb  # (batch_size, num_negatives, embedding_dim)
)  # (batch_size, num_negatives)

# Loss handles multiple negatives automatically
loss = margin_loss(pos_scores, neg_scores)
```

### Hierarchical Group Aggregation

```python
from KGAG.aggregators import HierarchicalGroupAggregator

# For large groups with diverse preferences
hierarchical_agg = HierarchicalGroupAggregator(
    embedding_dim=64,
    num_subgroups=3
)

# Get group embedding with subgroup information
group_emb, subgroup_info = hierarchical_agg(member_embeddings, item_embedding)

# Analyze subgroup structure
print("Subgroup assignments:", subgroup_info['subgroup_assignments'])
print("Subgroup attention:", subgroup_info['subgroup_attention'])
```

## 📊 Examples

Run the example file to see all features in action:

```bash
python KGAG/example_usage.py
```

Examples included:
1. **Basic Usage**: Model initialization and forward pass
2. **Group Aggregation**: Attention mechanism and member influence
3. **Loss Functions**: Different loss types and components
4. **Training Loop**: Complete training pipeline
5. **Aggregator Comparison**: Compare different strategies

## 🎓 Paper Reference

This implementation is based on the paper:
**"Knowledge-Aware Group Representation Learning for Group Recommendation"**

Key contributions implemented:
1. ✅ Knowledge Graph Integration via Collaborative Knowledge Graph (CKG)
2. ✅ Attention-based Group Aggregation (paper SP + PI)
3. ✅ Margin-based Ranking Loss for discriminative embeddings

## 📝 Parameters

### KGAG Model
- `num_users`: Number of users
- `num_entities`: Number of entities (items + KG entities)
- `num_relations`: Number of relation types in KG
- `embedding_dim`: Dimension of embeddings (default: 64)
- `num_layers`: Number of GCN layers (default: 3)
- `aggregation_type`: GCN aggregation ('bi-interaction', 'gcn', 'graphsage')

### AttentionGroupAggregator
- `embedding_dim`: Embedding dimension
- `dropout`: Dropout probability (default: 0.1)

### MarginRankingLoss
- `margin`: Minimum score difference (default: 1.0)
- `reduction`: Loss reduction ('mean', 'sum', 'none')

### CombinedLoss
- `ranking_loss`: Loss type ('margin', 'bpr', 'adaptive', 'contrastive')
- `margin`: Margin value for margin-based losses
- `reg_weight`: L2 regularization weight
- `kg_weight`: Knowledge graph loss weight

## 🔍 Tips

- **Margin tuning**: Start with margin=1.0, increase for stricter separation
- **Regularization**: Adjust `reg_weight` (1e-5 to 1e-4) to prevent overfitting
- **Multiple negatives**: Use 3-5 negatives per positive for better training
- **Learning rate**: Start with 0.001, decrease if loss oscillates

## 🐛 Troubleshooting

**Issue**: Attention weights too uniform
- **Solution**: Increase dropout, tune attention network architecture

**Issue**: Loss not decreasing
- **Solution**: Check margin value, try smaller margin or different loss type

**Issue**: Overfitting
- **Solution**: Increase regularization weight, add dropout

**Issue**: Out of memory
- **Solution**: Reduce batch size, use fewer GCN layers, or smaller embeddings

## 📄 License

See parent directory LICENSE file.
