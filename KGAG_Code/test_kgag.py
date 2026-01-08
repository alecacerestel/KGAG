"""
Unit tests for KGAG module components

Run with: python -m pytest KGAG/test_kgag.py
or: python KGAG/test_kgag.py
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KGAG import (
    KGAG,
    GCNLayer,
    AttentionGroupAggregator,
    SimpleGroupAggregator,
    HierarchicalGroupAggregator,
    MarginRankingLoss,
    BPRLoss,
    AdaptiveMarginLoss,
    CombinedLoss,
    TransELoss
)


def test_gcn_layer():
    """Test GCN layer forward pass"""
    print("Testing GCN Layer...")
    
    in_dim = 32
    out_dim = 64
    num_nodes = 100
    
    # Create GCN layer
    gcn = GCNLayer(in_dim, out_dim, aggregation_type='bi-interaction')
    
    # Create dummy data
    ego_embeddings = torch.randn(num_nodes, in_dim)
    
    # Create sparse adjacency matrix
    num_edges = 200
    edge_indices = torch.randint(0, num_nodes, (2, num_edges))
    edge_values = torch.ones(num_edges)
    adjacency_matrix = torch.sparse_coo_tensor(
        indices=edge_indices,
        values=edge_values,
        size=(num_nodes, num_nodes)
    )
    
    # Forward pass
    output = gcn(ego_embeddings, adjacency_matrix)
    
    assert output.shape == (num_nodes, out_dim), f"Expected shape ({num_nodes}, {out_dim}), got {output.shape}"
    print("✓ GCN Layer test passed!")
    return True


def test_kgag_model():
    """Test KGAG model initialization and forward pass"""
    print("\nTesting KGAG Model...")
    
    num_users = 50
    num_entities = 200
    num_relations = 5
    embedding_dim = 32
    batch_size = 16
    
    # Initialize model
    model = KGAG(
        num_users=num_users,
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=2
    )
    
    # Create dummy edges
    user_item_edges = torch.stack([
        torch.randint(0, num_users, (100,)),
        torch.randint(0, num_entities // 2, (100,))
    ])
    
    kg_edges = torch.stack([
        torch.randint(0, num_entities, (300,)),
        torch.randint(0, num_entities, (300,))
    ])
    
    # Create batch
    user_indices = torch.randint(0, num_users, (batch_size,))
    item_indices = torch.randint(0, num_entities // 2, (batch_size,))
    
    # Forward pass
    user_embeddings, item_embeddings = model(
        user_indices, item_indices,
        user_item_edges, kg_edges
    )
    
    assert user_embeddings.shape == (batch_size, embedding_dim)
    assert item_embeddings.shape == (batch_size, embedding_dim)
    print("✓ KGAG Model test passed!")
    return True


def test_attention_aggregator():
    """Test attention-based group aggregator"""
    print("\nTesting Attention Aggregator...")
    
    embedding_dim = 64
    num_members = 5
    
    # Create aggregator
    aggregator = AttentionGroupAggregator(
        embedding_dim=embedding_dim,
        attention_type='concat'
    )
    
    # Create dummy data
    member_embeddings = torch.randn(num_members, embedding_dim)
    item_embedding = torch.randn(embedding_dim)
    
    # Forward pass
    group_embedding, attention_weights = aggregator(member_embeddings, item_embedding)
    
    assert group_embedding.shape == (embedding_dim,)
    assert attention_weights.shape == (num_members,)
    assert torch.abs(attention_weights.sum() - 1.0) < 1e-5, "Attention weights should sum to 1"
    print(f"  Attention weights: {attention_weights.detach().numpy()}")
    print("✓ Attention Aggregator test passed!")
    return True


def test_simple_aggregator():
    """Test simple aggregators (mean, max, min)"""
    print("\nTesting Simple Aggregators...")
    
    embedding_dim = 64
    num_members = 5
    member_embeddings = torch.randn(num_members, embedding_dim)
    
    # Test mean aggregator
    agg_mean = SimpleGroupAggregator(aggregation_type='mean')
    group_emb, _ = agg_mean(member_embeddings)
    assert group_emb.shape == (embedding_dim,)
    
    # Test max aggregator
    agg_max = SimpleGroupAggregator(aggregation_type='max')
    group_emb, _ = agg_max(member_embeddings)
    assert group_emb.shape == (embedding_dim,)
    
    # Test min aggregator
    agg_min = SimpleGroupAggregator(aggregation_type='min')
    group_emb, _ = agg_min(member_embeddings)
    assert group_emb.shape == (embedding_dim,)
    
    print("✓ Simple Aggregators test passed!")
    return True


def test_hierarchical_aggregator():
    """Test hierarchical group aggregator"""
    print("\nTesting Hierarchical Aggregator...")
    
    embedding_dim = 64
    num_members = 10
    num_subgroups = 3
    
    # Create aggregator
    aggregator = HierarchicalGroupAggregator(
        embedding_dim=embedding_dim,
        num_subgroups=num_subgroups,
        attention_type='concat'
    )
    
    # Create dummy data
    member_embeddings = torch.randn(num_members, embedding_dim)
    item_embedding = torch.randn(embedding_dim)
    
    # Forward pass
    group_embedding, subgroup_info = aggregator(member_embeddings, item_embedding)
    
    assert group_embedding.shape == (embedding_dim,)
    assert 'subgroup_assignments' in subgroup_info
    assert 'subgroup_attention' in subgroup_info
    assert subgroup_info['subgroup_assignments'].shape == (num_members, num_subgroups)
    
    print("✓ Hierarchical Aggregator test passed!")
    return True


def test_margin_loss():
    """Test margin ranking loss"""
    print("\nTesting Margin Ranking Loss...")
    
    batch_size = 32
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size)
    
    # Test with single negative
    loss_fn = MarginRankingLoss(margin=1.0)
    loss = loss_fn(pos_scores, neg_scores)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test with multiple negatives
    num_negatives = 5
    neg_scores_multi = torch.randn(batch_size, num_negatives)
    loss_multi = loss_fn(pos_scores, neg_scores_multi)
    assert loss_multi.dim() == 0
    assert loss_multi.item() >= 0
    
    print(f"  Loss (single neg): {loss.item():.4f}")
    print(f"  Loss (multi neg): {loss_multi.item():.4f}")
    print("✓ Margin Ranking Loss test passed!")
    return True


def test_bpr_loss():
    """Test BPR loss"""
    print("\nTesting BPR Loss...")
    
    batch_size = 32
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size)
    
    loss_fn = BPRLoss()
    loss = loss_fn(pos_scores, neg_scores)
    
    assert loss.dim() == 0
    assert loss.item() >= 0
    print(f"  BPR Loss: {loss.item():.4f}")
    print("✓ BPR Loss test passed!")
    return True


def test_adaptive_margin_loss():
    """Test adaptive margin loss"""
    print("\nTesting Adaptive Margin Loss...")
    
    batch_size = 32
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size)
    
    loss_fn = AdaptiveMarginLoss(base_margin=1.0, adaptive_factor=0.5)
    loss = loss_fn(pos_scores, neg_scores)
    
    assert loss.dim() == 0
    assert loss.item() >= 0
    print(f"  Adaptive Margin Loss: {loss.item():.4f}")
    print("✓ Adaptive Margin Loss test passed!")
    return True


def test_combined_loss():
    """Test combined loss with regularization"""
    print("\nTesting Combined Loss...")
    
    batch_size = 32
    embedding_dim = 64
    
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size)
    user_embeddings = torch.randn(batch_size, embedding_dim)
    item_embeddings = torch.randn(batch_size, embedding_dim)
    
    loss_fn = CombinedLoss(
        ranking_loss='margin',
        margin=1.0,
        reg_weight=1e-4
    )
    
    loss, loss_dict = loss_fn(
        pos_scores, neg_scores,
        embeddings=[user_embeddings, item_embeddings]
    )
    
    assert loss.dim() == 0
    assert loss.item() >= 0
    assert 'total' in loss_dict
    assert 'ranking' in loss_dict
    assert 'regularization' in loss_dict
    
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Ranking loss: {loss_dict['ranking']:.4f}")
    print(f"  Regularization: {loss_dict['regularization']:.6f}")
    print("✓ Combined Loss test passed!")
    return True


def test_transe_loss():
    """Test TransE loss for KG embeddings"""
    print("\nTesting TransE Loss...")
    
    batch_size = 32
    embedding_dim = 64
    
    # Positive triple
    pos_head = torch.randn(batch_size, embedding_dim)
    pos_relation = torch.randn(batch_size, embedding_dim)
    pos_tail = torch.randn(batch_size, embedding_dim)
    
    # Negative triple
    neg_head = torch.randn(batch_size, embedding_dim)
    neg_relation = torch.randn(batch_size, embedding_dim)
    neg_tail = torch.randn(batch_size, embedding_dim)
    
    loss_fn = TransELoss(margin=1.0, norm='L2')
    loss = loss_fn(pos_head, pos_relation, pos_tail,
                   neg_head, neg_relation, neg_tail)
    
    assert loss.dim() == 0
    assert loss.item() >= 0
    print(f"  TransE Loss: {loss.item():.4f}")
    print("✓ TransE Loss test passed!")
    return True


def test_group_embedding():
    """Test group embedding generation"""
    print("\nTesting Group Embedding Generation...")
    
    num_users = 50
    num_entities = 200
    embedding_dim = 32
    
    # Initialize model
    model = KGAG(
        num_users=num_users,
        num_entities=num_entities,
        num_relations=5,
        embedding_dim=embedding_dim,
        num_layers=2
    )
    
    # Create dummy edges
    user_item_edges = torch.stack([
        torch.randint(0, num_users, (100,)),
        torch.randint(0, num_entities // 2, (100,))
    ])
    kg_edges = torch.stack([
        torch.randint(0, num_entities, (300,)),
        torch.randint(0, num_entities, (300,))
    ])
    
    # Define group and items
    group_members = [5, 12, 23, 31]
    candidate_items = torch.tensor([10, 25, 50])
    
    # Get group embeddings
    group_embeddings, attention_weights = model.get_group_embedding(
        group_members=group_members,
        item_indices=candidate_items,
        user_item_edges=user_item_edges,
        kg_edges=kg_edges
    )
    
    assert group_embeddings.shape == (len(candidate_items), embedding_dim)
    assert attention_weights.shape == (len(candidate_items), len(group_members))
    
    print(f"  Group embeddings shape: {group_embeddings.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Sample attention: {attention_weights[0].detach().numpy()}")
    print("✓ Group Embedding Generation test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running KGAG Module Tests")
    print("=" * 80)
    
    tests = [
        ("GCN Layer", test_gcn_layer),
        ("KGAG Model", test_kgag_model),
        ("Attention Aggregator", test_attention_aggregator),
        ("Simple Aggregators", test_simple_aggregator),
        ("Hierarchical Aggregator", test_hierarchical_aggregator),
        ("Margin Ranking Loss", test_margin_loss),
        ("BPR Loss", test_bpr_loss),
        ("Adaptive Margin Loss", test_adaptive_margin_loss),
        ("Combined Loss", test_combined_loss),
        ("TransE Loss", test_transe_loss),
        ("Group Embedding", test_group_embedding),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} test FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All tests passed successfully!")
    else:
        print(f"⚠️  {failed} test(s) failed. Please review.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
