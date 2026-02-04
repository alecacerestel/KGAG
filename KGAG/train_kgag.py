"""
Minimal Training Script for KGAG Model with BeHAVE Datasets


Usage:
    python train_kgag.py --data_dir dataset/MovieLens_RecBole_KG/BeHAVE/E_AU
"""

import sys
from pathlib import Path

import torch
import torch.optim as optim

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from KGAG import KGAG, MarginRankingLoss
from KGAG.dataloader import load_behave_data


def train(model, loader, user_item_edges, kg_edges, loss_fn, optimizer,
          batch_size, num_negatives, device):
    """Train KGAG for one epoch following the paper's group-level objective.

    Paper training objective (simplified):
        - For each observed group–item interaction (g, v) and sampled negative
          item v⁻, enforce s(g, v) > s(g, v⁻) + margin.
        - s(g, v) is computed from:
            1) CKG-based user/item embeddings
            2) SP + PI attention to aggregate members into a group embedding
            3) Inner-product between group and item embeddings.

    This function:
        - Uses group–item interactions from BeHAVE (loader.group_item_interactions)
        - Uses group membership (loader.get_group_members())
        - Uses KGAG's AttentionGroupAggregator (SP + PI) to get group embeddings
        - Applies margin ranking loss on group–item scores.

    Returns:
        Average loss over all batches.
    """

    model.train()
    total_loss = 0.0
    num_batches = 0

    # ------------------------------------------------------------------
    # 1) Prepare mappings and training samples
    # ------------------------------------------------------------------
    # Negative sampling in loader is based on group–item interactions.
    # What is this
    pos_samples_internal, neg_samples_entity = loader.get_training_samples(
        num_negatives=num_negatives
    )

    # Build mapping: internal item index -> entity index
    # item_id_map: original_item_id -> internal_item_idx
    # entity_id_map: original_item_id -> entity_idx
    internal_to_entity = {}
    for original_item_id, internal_idx in loader.item_id_map.items():
        entity_idx = loader.entity_id_map[original_item_id]
        internal_to_entity[internal_idx] = entity_idx

    # Convert positive samples to entity space (g, v_entity)
    pos_samples = []
    for group_id, internal_item in pos_samples_internal:
        entity_idx = internal_to_entity[internal_item]
        pos_samples.append((group_id, entity_idx))

    # Negative samples are already in entity space (g, v_entity)
    # Ensure we have a one-to-one pairing (assume num_negatives = 1)
    if len(neg_samples_entity) != len(pos_samples):
        neg_samples = neg_samples_entity[: len(pos_samples)]
    else:
        neg_samples = neg_samples_entity

    # Group membership: group_id -> list[user_id]
    # What is this
    group_members = loader.get_group_members()

    # ------------------------------------------------------------------
    # 2) Compute CKG-based user/item embeddings once per epoch
    # ------------------------------------------------------------------
    # This matches the paper's flow: first learn user/item representations
    # from the collaborative knowledge graph, then apply attention to
    # aggregate members into group embeddings.
    user_item_edges = user_item_edges.to(device)
    kg_edges = kg_edges.to(device)

    # What is this
    all_user_embeddings, all_entity_embeddings = model.get_all_embeddings(
        user_item_edges, kg_edges
    )

    # ------------------------------------------------------------------
    # 3) Shuffle and create batches over group–item pairs
    # ------------------------------------------------------------------
    num_samples = len(pos_samples)
    indices = torch.randperm(num_samples, device=device)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_group_emb_pos = []
        batch_item_emb_pos = []
        batch_group_emb_neg = []
        batch_item_emb_neg = []

        # --------------------------------------------------------------
        # For each (group, pos_item, neg_item) in batch:
        #   - get member embeddings from all_user_embeddings
        #   - get item embeddings from all_entity_embeddings
        #   - aggregate members via SP + PI attention to group embedding
        # --------------------------------------------------------------
        # GROUP LOSS 
        # ------------
        # For batch in group-item interactions:
        for idx in batch_indices.tolist():
            group_id_pos, pos_entity = pos_samples[idx]
            group_id_neg, neg_entity = neg_samples[idx]

            # Sanity: group ids should match for pos and neg
            assert group_id_pos == group_id_neg
            group_id = group_id_pos

            # Member indices for this group (internal user ids)
            members = group_members[group_id]
            if isinstance(members, list):
                members = torch.tensor(members, dtype=torch.long, device=device)
            else:
                members = members.to(device)

            # Consider using: model.user_embedding(members)
            member_embeds = all_user_embeddings[members]  # (num_members, d)

            # Item embeddings for positive and negative items
            # Consider using: model.entity_embedding(pos_entity)
            # Consider using: model.entity_embedding(neg_entity)
            pos_item_embed = all_entity_embeddings[pos_entity]
            neg_item_embed = all_entity_embeddings[neg_entity]

            # Group embeddings via SP + PI attention
            group_emb_pos, _ = model.group_aggregator(member_embeds, pos_item_embed)
            group_emb_neg, _ = model.group_aggregator(member_embeds, neg_item_embed)

            batch_group_emb_pos.append(group_emb_pos)
            batch_item_emb_pos.append(pos_item_embed)
            batch_group_emb_neg.append(group_emb_neg)
            batch_item_emb_neg.append(neg_item_embed)

        # Stack to tensors of shape (batch_size, embedding_dim)
        batch_group_emb_pos = torch.stack(batch_group_emb_pos)
        batch_item_emb_pos = torch.stack(batch_item_emb_pos)
        batch_group_emb_neg = torch.stack(batch_group_emb_neg)
        batch_item_emb_neg = torch.stack(batch_item_emb_neg)

        # ------------------------------------------------------------------
        # 4) Compute scores and margin ranking loss (group-level)
        # ------------------------------------------------------------------
        pos_scores = model.predict(batch_group_emb_pos, batch_item_emb_pos)
        neg_scores = model.predict(batch_group_emb_neg, batch_item_emb_neg)

        loss = loss_fn(pos_scores, neg_scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # # GROUP ITEM LOSS
        # group_loss = 0.0 
        # group_batches = loader.get_batches(batch_size, mode='group')
        
        # for (group_ids, pos_items, neg_items) in group_batches:
        #     group_members = loader.get_group_members()[group_ids]
            
        
        # USER ITEM LOSS (optional)
        # -------------------------
        # for batch in user-item interactions:
        user_loss = 0.0 
        user_batches = loader.get_batches(batch_size, mode='user')
        for (user_ids, pos_items, neg_items) in user_batches:

            user_embeds = model.user_embedding(user_ids.to(device))
            pos_item_embeds = model.entity_embedding(pos_items.to(device))
            neg_item_embeds = model.entity_embedding(neg_items.to(device))

            pos_scores_ui = model.predict(user_embeds, pos_item_embeds)
            neg_scores_ui = model.predict(user_embeds, neg_item_embeds)

            loss_ui = loss_fn(pos_scores_ui, neg_scores_ui)
            optimizer.zero_grad()
            loss_ui.backward()
            optimizer.step()
            user_loss += loss_ui.item()
            
        user_loss = user_loss / (len(user_batches) + 1e-10)

        total_loss += loss.item() +  user_loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    # Configuration - modify these based on your paper/experiments
    DATA_DIR = 'dataset/MovieLens_RecBole_KG/BeHAVE/E_AU'
    EPOCHS = 20
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 64
    NUM_LAYERS = 3
    MARGIN = 1.0  # From paper: margin ranking loss
    NUM_NEGATIVES = 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load BeHAVE data
    print(f"\nLoading data from {DATA_DIR}...")
    loader = load_behave_data(DATA_DIR)
    user_item_edges = loader.get_user_item_edges()
    kg_edges = loader.get_kg_edges()
    
    print(f"Users: {loader.num_users}, Items: {loader.num_items}, Entities: {loader.num_entities}")
    print(f"User-Item edges: {user_item_edges.shape[1]}, KG edges: {kg_edges.shape[1]}")
    
    # Initialize KGAG model
    print("\nInitializing KGAG model...")
    model = KGAG(
        num_users=loader.num_users,
        num_entities=loader.num_entities,
        num_relations=loader.num_relations,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Loss and optimizer
    loss_fn = MarginRankingLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Trainding loop
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        avg_loss = train(
            model, loader, user_item_edges, kg_edges,
            loss_fn, optimizer, BATCH_SIZE, NUM_NEGATIVES, device
        )
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = 'kgag_trained.pt'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    main()

