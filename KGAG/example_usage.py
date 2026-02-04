"""
Example Usage of KGAG Model with Group Aggregators and Loss Functions

This script demonstrates how to use the KGAG model for group recommendation
with attention-based aggregation and margin ranking loss.
"""

import os
import sys

# Ensure the project root is on PYTHONPATH when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.optim as optim
from KGAG import (
    KGAG,
    AttentionGroupAggregator,
    SimpleGroupAggregator,
    MarginRankingLoss,
    BPRLoss,
    CombinedLoss
)


def example_basic_usage():
    """
    Basic example: Initialize model and perform forward pass
    """
    print("=" * 80)
    print("Example 1: Basic KGAG Model Usage")
    print("=" * 80)
    
    # Model parameters
    num_users = 100
    num_entities = 500  # Includes items and KG entities
    num_relations = 10
    embedding_dim = 64
    num_layers = 3
    
    # Initialize KGAG model
    model = KGAG(
        num_users=num_users,
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        aggregation_type='bi-interaction'
    )
    
    print(f"Model initialized with:")
    print(f"  - Users: {num_users}")
    print(f"  - Entities: {num_entities}")
    print(f"  - Relations: {num_relations}")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - GCN layers: {num_layers}")
    
    # Dummy data: User-Item interactions
    num_ui_edges = 200
    user_item_edges = torch.stack([
        torch.randint(0, num_users, (num_ui_edges,)),  # User indices
        torch.randint(0, num_entities // 2, (num_ui_edges,))  # Item indices (first half of entities)
    ])
    
    # Dummy data: Knowledge Graph edges
    num_kg_edges = 1000
    kg_edges = torch.stack([
        torch.randint(0, num_entities, (num_kg_edges,)),  # Head entities
        torch.randint(0, num_entities, (num_kg_edges,))   # Tail entities
    ])
    
    # Batch data
    batch_size = 32
    user_indices = torch.randint(0, num_users, (batch_size,))
    item_indices = torch.randint(0, num_entities // 2, (batch_size,))
    
    # Forward pass
    user_embeddings, item_embeddings = model(
        user_indices=user_indices,
        item_indices=item_indices,
        user_item_edges=user_item_edges,
        kg_edges=kg_edges
    )
    
    print(f"\nForward pass completed:")
    print(f"  - User embeddings shape: {user_embeddings.shape}")
    print(f"  - Item embeddings shape: {item_embeddings.shape}")
    
    # Compute scores
    scores = model.predict(user_embeddings, item_embeddings)
    print(f"  - Prediction scores shape: {scores.shape}")
    print(f"  - Sample scores: {scores[:5]}")
    
    return model, user_item_edges, kg_edges


def example_group_aggregation():
    """
    Example: Group embedding with attention mechanism
    """
    print("\n" + "=" * 80)
    print("Example 2: Group Aggregation with Attention")
    print("=" * 80)
    
    # Initialize model
    num_users = 50
    num_entities = 200
    embedding_dim = 64
    
    model = KGAG(
        num_users=num_users,
        num_entities=num_entities,
        num_relations=5,
        embedding_dim=embedding_dim,
        num_layers=2
    )
    
    # Dummy edges
    user_item_edges = torch.stack([
        torch.randint(0, num_users, (100,)),
        torch.randint(0, num_entities // 2, (100,))
    ])
    kg_edges = torch.stack([
        torch.randint(0, num_entities, (300,)),
        torch.randint(0, num_entities, (300,))
    ])
    
    # Define a group
    group_members = [5, 12, 23, 31]  # 4 members
    print(f"Group members: {group_members}")
    
    # Items to recommend
    candidate_items = torch.tensor([10, 25, 50])
    print(f"Candidate items: {candidate_items.tolist()}")
    
    # Get group embeddings with attention
    group_embeddings, attention_weights = model.get_group_embedding(
        group_members=group_members,
        item_indices=candidate_items,
        user_item_edges=user_item_edges,
        kg_edges=kg_edges
    )
    
    print(f"\nGroup aggregation results:")
    print(f"  - Group embeddings shape: {group_embeddings.shape}")
    print(f"  - Attention weights shape: {attention_weights.shape}")
    
    # Show attention weights for each item
    for i, item_id in enumerate(candidate_items):
        print(f"\n  Item {item_id.item()}:")
        print(f"    Attention weights: {attention_weights[i].detach().numpy()}")
        print(f"    Most influential member: Member {group_members[attention_weights[i].argmax()]}")
    
    # Compute group-item scores
    all_item_embeddings = model.get_all_embeddings(user_item_edges, kg_edges)[1]
    item_embeddings = all_item_embeddings[candidate_items]
    scores = (group_embeddings * item_embeddings).sum(dim=1)
    
    print(f"\n  Group-item scores: {scores.detach().numpy()}")
    print(f"  Best recommendation: Item {candidate_items[scores.argmax()].item()}")
    
    return model


def example_loss_functions():
    """
    Example: Different loss functions for training
    """
    print("\n" + "=" * 80)
    print("Example 3: Loss Functions")
    print("=" * 80)
    
    # Dummy scores
    batch_size = 16
    pos_scores = torch.randn(batch_size)  # Positive scores
    neg_scores = torch.randn(batch_size)  # Negative scores
    
    print(f"Batch size: {batch_size}")
    print(f"Positive scores (sample): {pos_scores[:3].detach().numpy()}")
    print(f"Negative scores (sample): {neg_scores[:3].detach().numpy()}")
    
    # 1. Margin Ranking Loss
    print("\n1. Margin Ranking Loss (margin=1.0)")
    margin_loss = MarginRankingLoss(margin=1.0)
    loss_value = margin_loss(pos_scores, neg_scores)
    print(f"   Loss value: {loss_value.item():.4f}")
    
    # 2. BPR Loss
    print("\n2. BPR Loss")
    bpr_loss = BPRLoss()
    loss_value = bpr_loss(pos_scores, neg_scores)
    print(f"   Loss value: {loss_value.item():.4f}")
    
    # 3. Combined Loss with regularization
    print("\n3. Combined Loss (Margin + L2 Regularization)")
    combined_loss = CombinedLoss(
        ranking_loss='margin',
        margin=1.0,
        reg_weight=1e-4
    )
    
    # Dummy embeddings for regularization
    user_embeddings = torch.randn(batch_size, 64)
    item_embeddings = torch.randn(batch_size, 64)
    
    loss_value, loss_dict = combined_loss(
        pos_scores, neg_scores,
        embeddings=[user_embeddings, item_embeddings]
    )
    
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   Ranking loss: {loss_dict['ranking']:.4f}")
    print(f"   Regularization: {loss_dict['regularization']:.4f}")
    
    # 4. Multiple negatives
    print("\n4. Margin Loss with Multiple Negatives")
    num_negatives = 5
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size, num_negatives)
    
    margin_loss = MarginRankingLoss(margin=1.0)
    loss_value = margin_loss(pos_scores, neg_scores)
    print(f"   Loss value with {num_negatives} negatives: {loss_value.item():.4f}")


def example_training_loop():
    """
    Example: Simple training loop with KGAG
    """
    print("\n" + "=" * 80)
    print("Example 4: Training Loop")
    print("=" * 80)
    
    # Initialize model
    model = KGAG(
        num_users=100,
        num_entities=500,
        num_relations=10,
        embedding_dim=64,
        num_layers=2
    )
    
    # Prepare data
    user_item_edges = torch.stack([
        torch.randint(0, 100, (200,)),
        torch.randint(0, 250, (200,))
    ])
    kg_edges = torch.stack([
        torch.randint(0, 500, (1000,)),
        torch.randint(0, 500, (1000,))
    ])
    
    # Loss and optimizer
    loss_fn = CombinedLoss(
        ranking_loss='margin',
        margin=1.0,
        reg_weight=1e-5
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    batch_size = 32
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 10
        
        for batch_idx in range(num_batches):
            # Sample batch data
            user_indices = torch.randint(0, 100, (batch_size,))
            pos_item_indices = torch.randint(0, 250, (batch_size,))
            neg_item_indices = torch.randint(0, 250, (batch_size,))
            
            # Forward pass
            user_embeddings_pos, item_embeddings_pos = model(
                user_indices, pos_item_indices,
                user_item_edges, kg_edges
            )
            user_embeddings_neg, item_embeddings_neg = model(
                user_indices, neg_item_indices,
                user_item_edges, kg_edges
            )
            
            # Compute scores
            pos_scores = model.predict(user_embeddings_pos, item_embeddings_pos)
            neg_scores = model.predict(user_embeddings_neg, item_embeddings_neg)
            
            # Compute loss
            loss, loss_dict = loss_fn(
                pos_scores, neg_scores,
                embeddings=[user_embeddings_pos, item_embeddings_pos]
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return model


def example_aggregator_comparison():
    """
    Example: Compare different aggregation strategies
    """
    print("\n" + "=" * 80)
    print("Example 5: Compare Aggregation Strategies")
    print("=" * 80)
    
    embedding_dim = 64
    num_members = 5
    
    # Create dummy member embeddings
    member_embeddings = torch.randn(num_members, embedding_dim)
    item_embedding = torch.randn(embedding_dim)
    
    print(f"Group with {num_members} members")
    print(f"Embedding dimension: {embedding_dim}")
    
    # 1. Attention-based aggregation (paper SP + PI)
    print("\n1. Attention Aggregation (paper SP + PI)")
    agg_attention = AttentionGroupAggregator(embedding_dim)
    group_emb, attn_weights = agg_attention(member_embeddings, item_embedding)
    print(f"   Attention weights: {attn_weights.detach().numpy()}")
    print(f"   Group embedding norm: {torch.norm(group_emb).item():.4f}")
    
    # 2. Simple mean aggregation
    print("\n2. Simple Mean Aggregation")
    agg_mean = SimpleGroupAggregator(aggregation_type='mean')
    group_emb, _ = agg_mean(member_embeddings, item_embedding)
    print(f"   Group embedding norm: {torch.norm(group_emb).item():.4f}")
    
    # 3. Simple max aggregation
    print("\n3. Simple Max Aggregation")
    agg_max = SimpleGroupAggregator(aggregation_type='max')
    group_emb, _ = agg_max(member_embeddings, item_embedding)
    print(f"   Group embedding norm: {torch.norm(group_emb).item():.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KGAG Model Examples - Group Aggregators and Loss Functions")
    print("=" * 80)
    
    # Run examples
    example_basic_usage()
    example_group_aggregation()
    example_loss_functions()
    example_training_loop()
    example_aggregator_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
