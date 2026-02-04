"""
Training Script for KGAG Model

Simple training loop for Knowledge-Aware Group Recommendation.

Usage:
    python KGAG/train.py --data_dir dataset/MovieLens_RecBole_KG/BeHAVE/S_AU/
    change to 50CU50DU_20G, data recover from the drive, 
    so I can use: C:/Python312/python.exe KGAG/train.py --data_dir dataset/50CU50DU_20G/ --epochs 2 --batch_size 10
    to do a short test run
"""

import os
import argparse
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataloader import BeHAVEDataLoader
from dataset import create_dataloaders
from model import KGAG
from losses import MarginRankingLoss


def train_epoch(model, train_loader, optimizer, loss_fn, device, user_item_edges, kg_edges, group_members):
    """
    Train for one epoch.
    
    Args:
        model: KGAG model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device (cuda or cpu)
        user_item_edges: User-item interaction edges
        kg_edges: Knowledge graph edges
        group_members: Dictionary mapping group_id to list of member user_ids
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for group_ids, pos_items, neg_items in tqdm(train_loader, desc="Training"):
        # Move to device
        group_ids = group_ids.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        
        batch_size = group_ids.size(0)
        pos_scores_list = []
        neg_scores_list = []
        
        # Process each group in the batch
        for i in range(batch_size):
            group_id = group_ids[i].item()
            members = torch.tensor(group_members[group_id], dtype=torch.long, device=device)
            
            # Get group embeddings for positive and negative items
            pos_group_emb, _ = model.get_group_embedding(members, pos_items[i:i+1], user_item_edges, kg_edges)
            neg_group_emb, _ = model.get_group_embedding(members, neg_items[i:i+1], user_item_edges, kg_edges)
            
            # Get item embeddings
            _, all_item_embeddings = model.get_all_embeddings(user_item_edges, kg_edges)
            pos_item_emb = all_item_embeddings[pos_items[i:i+1]]
            neg_item_emb = all_item_embeddings[neg_items[i:i+1]]
            
            # Calculate scores
            pos_score = model.predict(pos_group_emb, pos_item_emb)
            neg_score = model.predict(neg_group_emb, neg_item_emb)
            
            pos_scores_list.append(pos_score)
            neg_scores_list.append(neg_score)
        
        pos_scores = torch.cat(pos_scores_list)
        neg_scores = torch.cat(neg_scores_list)
        
        # Compute loss
        loss = loss_fn(pos_scores, neg_scores)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, test_loader, device, k=10):
    """
    Evaluate model on test set.
    
    Args:
        model: KGAG model
        test_loader: Test data loader
        device: Device
        k: Top-K for Hit@K metric
        
    Returns:
        hit_rate: Hit@K score
    """
    model.eval()
    hits = 0
    total = 0
    
    with torch.no_grad():
        for group_ids, true_items, candidates in tqdm(test_loader, desc="Evaluating"):
            batch_size = group_ids.size(0)
            
            for i in range(batch_size):
                group_id = group_ids[i]
                true_item = true_items[i]
                cand_items = candidates[i]
                
                # Score all candidates
                group_tensor = group_id.repeat(len(cand_items)).to(device)
                items_tensor = torch.tensor(cand_items).to(device)
                scores = model(group_tensor, items_tensor)
                
                # Get top-K
                _, top_k_indices = torch.topk(scores, k)
                top_k_items = [cand_items[idx] for idx in top_k_indices.cpu().numpy()]
                
                # Check if true item is in top-K
                if true_item.item() in top_k_items:
                    hits += 1
                total += 1
    
    hit_rate = hits / total if total > 0 else 0
    return hit_rate


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("KGAG Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    data_loader = BeHAVEDataLoader(args.data_dir)
    data_loader.load_all()
    
    # Get edge information for the graph
    user_item_edges = data_loader.get_user_item_edges().to(device)
    kg_edges = data_loader.get_kg_edges().to(device)
    group_members = data_loader.get_group_members()
    print(f"User-item edges: {user_item_edges.shape}")
    print(f"KG edges: {kg_edges.shape}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        data_loader,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives
    )
    
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    # When there's no KG, entities = items
    num_entities = data_loader.num_entities if data_loader.num_entities > 0 else data_loader.num_items
    model = KGAG(
        num_users=data_loader.num_users,
        num_entities=num_entities,
        num_relations=data_loader.num_relations,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Load pre-trained embeddings if available
    embeddings = data_loader.load_item_embeddings()
    if embeddings is not None:
        print("Using pre-trained item embeddings")
        # TODO: Initialize model with embeddings
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MarginRankingLoss(margin=args.margin)
    
    # Training loop
    print("\nStarting training...")
    best_hit_rate = 0
    loss_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, user_item_edges, kg_edges, group_members)
        loss_history.append(avg_loss)
        print(f"Loss: {avg_loss:.4f}")
        
        # Evaluate
        if test_loader and epoch % args.eval_every == 0:
            hit_rate = evaluate(model, test_loader, device, k=args.top_k)
            print(f"Hit@{args.top_k}: {hit_rate:.4f}")
            
            # Save best model
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            # save every 5 epochs
            # where is save?
            # int_epoch_5.pt
            # int_epoch_10.pt
            # ...

            # how to charge the checkpoint?
            


    
    # Save loss history
    loss_path = os.path.join(args.output_dir, 'loss_history.npy')
    np.save(loss_path, loss_history)
    print(f"\nSaved loss history to {loss_path}")
    
    print("\nTraining complete!")
    if test_loader:
        print(f"Best Hit@{args.top_k}: {best_hit_rate:.4f}")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train KGAG model')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BeHAVE output directory')
    parser.add_argument('--output_dir', type=str, default='checkpoint/KGAG',
                        help='Directory to save checkpoints')
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GCN layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for margin loss')
    parser.add_argument('--num_negatives', type=int, default=1,
                        help='Number of negative samples per positive')
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--top_k', type=int, default=10,
                        help='K for Hit@K metric')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()
