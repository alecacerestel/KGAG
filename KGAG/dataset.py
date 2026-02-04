"""
PyTorch Dataset for KGAG Model Training

This dataset handles positive and negative sampling for group recommendation.
Each sample contains: (group_id, positive_item, negative_item)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class KGAGDataset(Dataset):
    """
    Dataset for KGAG model training.
    
    Generates training samples with positive and negative items for each group.
    
    Args:
        positive_samples: List of (group_id, item_id) tuples
        num_items: Total number of items
        num_negatives: Number of negative samples per positive sample
        
    Returns:
        (group_id, positive_item, negative_item) for each sample
    """
    
    def __init__(self, positive_samples, num_items, num_negatives=1):
        """
        Initialize dataset.
        
        Args:
            positive_samples: List of (group_id, item_id) positive interactions
            num_items: Total number of items in the catalog
            num_negatives: Negative samples per positive (default: 1)
        """
        self.positive_samples = positive_samples
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build set of positive items for each group
        self.group_positive_items = defaultdict(set)
        for group_id, item_id in positive_samples:
            self.group_positive_items[group_id].add(item_id)
        
        # Pre-generate all training samples
        self.samples = []
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate all training samples with negatives."""
        for group_id, pos_item in self.positive_samples:
            for _ in range(self.num_negatives):
                # Sample negative item
                neg_item = self._sample_negative(group_id)
                self.samples.append((group_id, pos_item, neg_item))
    
    def _sample_negative(self, group_id):
        """
        Sample a negative item for a group.
        
        Args:
            group_id: Group ID
            
        Returns:
            neg_item: Item ID that the group has not interacted with
        """
        group_positives = self.group_positive_items[group_id]
        
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in group_positives:
                return neg_item
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            group_id: Group ID (int)
            pos_item: Positive item ID (int)
            neg_item: Negative item ID (int)
        """
        group_id, pos_item, neg_item = self.samples[idx]
        return group_id, pos_item, neg_item


class KGAGTestDataset(Dataset):
    """
    Dataset for KGAG model evaluation.
    
    For each test interaction, generates candidate list:
    - The true positive item
    - K random negative items
    
    This is used for ranking metrics like Hit@K and NDCG@K.
    """
    
    def __init__(self, test_samples, num_items, num_candidates=100):
        """
        Initialize test dataset.
        
        Args:
            test_samples: List of (group_id, item_id) test interactions
            num_items: Total number of items
            num_candidates: Number of candidate items per test (default: 100)
        """
        self.test_samples = test_samples
        self.num_items = num_items
        self.num_candidates = num_candidates
        
        # Build positive items per group
        self.group_positive_items = defaultdict(set)
        for group_id, item_id in test_samples:
            self.group_positive_items[group_id].add(item_id)
    
    def __len__(self):
        """Return number of test samples."""
        return len(self.test_samples)
    
    def __getitem__(self, idx):
        """
        Get test sample with candidate items.
        
        Args:
            idx: Sample index
            
        Returns:
            group_id: Group ID
            true_item: The ground truth item
            candidates: List of candidate items (including true item)
        """
        group_id, true_item = self.test_samples[idx]
        
        # Generate negative candidates
        candidates = [true_item]
        group_positives = self.group_positive_items[group_id]
        
        while len(candidates) < self.num_candidates:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in group_positives:
                candidates.append(neg_item)
        
        return group_id, true_item, candidates


def create_dataloaders(data_loader, batch_size=256, num_negatives=1, num_workers=0):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        data_loader: BeHAVEDataLoader instance with loaded data
        batch_size: Batch size for training (default: 256)
        num_negatives: Negative samples per positive (default: 1)
        num_workers: Number of workers for data loading (default: 0)
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing (or None if no test data)
    """
    from torch.utils.data import DataLoader
    
    # Create training dataset
    train_dataset = KGAGDataset(
        positive_samples=data_loader.group_item_interactions,
        num_items=data_loader.num_items,
        num_negatives=num_negatives
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Create test dataset if available
    test_groups, _ = data_loader.load_test_data()
    test_loader = None
    
    if test_groups:
        test_dataset = KGAGTestDataset(
            test_samples=test_groups,
            num_items=data_loader.num_items,
            num_candidates=100
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return train_loader, test_loader
