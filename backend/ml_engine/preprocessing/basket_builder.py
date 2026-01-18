"""
Basket Builder - Preprocessing Module

Converts raw transaction data into ML-ready basket format
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import pickle
from pathlib import Path


class BasketBuilder:
    """
    Preprocesses transaction data into basket format for association rules
    """
    
    def __init__(self):
        self.category_map = {}  # item_id -> category
        self.transactions_df = None
        self.baskets = []
        self.sequences = []
    
    def load_data(self, transactions_path: str) -> pd.DataFrame:
        """
        Load transaction data from CSV
        
        Expected columns:
        - transaction_id
        - customer_id (optional)
        - timestamp
        - product_id
        - product_name
        - department
        - aisle
        - price
        """
        print(f" Loading data from {transactions_path}...")
        self.transactions_df = pd.read_csv(transactions_path)
        
        # Convert timestamp to datetime
        self.transactions_df['timestamp'] = pd.to_datetime(
            self.transactions_df['timestamp']
        )
        
        # Sort by transaction and timestamp
        self.transactions_df = self.transactions_df.sort_values(
            ['transaction_id', 'timestamp']
        )
        
        print(f" Loaded {len(self.transactions_df)} order lines")
        print(f"   Unique transactions: {self.transactions_df['transaction_id'].nunique()}")
        print(f"   Unique products: {self.transactions_df['product_name'].nunique()}")
        
        return self.transactions_df
    
    def build_category_map(self, level: str = 'department') -> Dict:
        """
        Create mapping from product to category
        
        Args:
            level: 'department' or 'aisle'
        """
        print(f"\n  Building category map at {level} level...")
        
        self.category_map = dict(
            zip(
                self.transactions_df['product_name'],
                self.transactions_df[level]
            )
        )
        
        unique_categories = self.transactions_df[level].nunique()
        print(f" Mapped {len(self.category_map)} products to {unique_categories} categories")
        
        return self.category_map
    
    def build_baskets(self, use_categories: bool = True) -> List[List[str]]:
        """
        Convert transactions into basket format
        
        Args:
            use_categories: If True, use categories instead of product names
        
        Returns:
            List of baskets, where each basket is a list of items
            Example: [['Rice', 'Dal', 'Oil'], ['Bread', 'Butter'], ...]
        """
        print("\n Building baskets...")
        
        # Group by transaction_id
        grouped = self.transactions_df.groupby('transaction_id')
        
        self.baskets = []
        
        for transaction_id, group in grouped:
            if use_categories:
                # Use departments/aisles
                basket = group['department'].unique().tolist()
            else:
                # Use product names
                basket = group['product_name'].unique().tolist()
            
            # Only keep baskets with 2+ items
            if len(basket) >= 2:
                self.baskets.append(basket)
        
        print(f" Built {len(self.baskets)} baskets")
        print(f"   Avg basket size: {np.mean([len(b) for b in self.baskets]):.2f}")
        print(f"   Min basket size: {min([len(b) for b in self.baskets])}")
        print(f"   Max basket size: {max([len(b) for b in self.baskets])}")
        
        return self.baskets
    
    def build_sequences(self, use_categories: bool = True) -> List[List[str]]:
        """
        Extract ordered sequences from transactions
        
        Args:
            use_categories: If True, use categories instead of product names
        
        Returns:
            List of sequences, where order matters
            Example: [['Vegetables', 'Rice', 'Dal'], ...]
        """
        print("\n Building sequences...")
        
        grouped = self.transactions_df.groupby('transaction_id')
        
        self.sequences = []
        
        for transaction_id, group in grouped:
            # Sort by timestamp within transaction
            group = group.sort_values('timestamp')
            
            if use_categories:
                sequence = group['department'].tolist()
            else:
                sequence = group['product_name'].tolist()
            
            # Remove consecutive duplicates (same category added twice)
            unique_sequence = []
            prev = None
            for item in sequence:
                if item != prev:
                    unique_sequence.append(item)
                    prev = item
            
            if len(unique_sequence) >= 2:
                self.sequences.append(unique_sequence)
        
        print(f" Built {len(self.sequences)} sequences")
        print(f"   Avg sequence length: {np.mean([len(s) for s in self.sequences]):.2f}")
        
        return self.sequences
    
    def get_basket_statistics(self) -> Dict:
        """
        Calculate statistics about baskets
        """
        if not self.baskets:
            raise ValueError("No baskets built yet. Run build_baskets() first.")
        
        basket_sizes = [len(b) for b in self.baskets]
        
        # Get basket values
        basket_values = []
        grouped = self.transactions_df.groupby('transaction_id')
        for transaction_id, group in grouped:
            basket_values.append(group['price'].sum())
        
        stats = {
            'num_baskets': len(self.baskets),
            'avg_basket_size': np.mean(basket_sizes),
            'median_basket_size': np.median(basket_sizes),
            'min_basket_size': min(basket_sizes),
            'max_basket_size': max(basket_sizes),
            'avg_basket_value': np.mean(basket_values),
            'median_basket_value': np.median(basket_values),
        }
        
        return stats
    
    def save(self, output_dir: str = "data"):
        """
        Save preprocessed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        data = {
            'baskets': self.baskets,
            'sequences': self.sequences,
            'category_map': self.category_map,
            'statistics': self.get_basket_statistics()
        }
        
        with open(output_path / "processed_baskets.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n Saved preprocessed data to {output_dir}/processed_baskets.pkl")
    
    def load(self, input_path: str = "data/processed_baskets.pkl"):
        """
        Load preprocessed data
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.baskets = data['baskets']
        self.sequences = data['sequences']
        self.category_map = data['category_map']
        
        print(f" Loaded preprocessed data from {input_path}")
        print(f"   Baskets: {len(self.baskets)}")
        print(f"   Sequences: {len(self.sequences)}")
        
        return data


# Example usage and testing
if __name__ == "__main__":
    # Initialize builder
    builder = BasketBuilder()
    
    # Load data
    transactions_df = builder.load_data("data/transactions.csv")
    
    # Build category map
    builder.build_category_map(level='department')
    
    # Build baskets (using departments)
    baskets = builder.build_baskets(use_categories=True)
    
    # Build sequences
    sequences = builder.build_sequences(use_categories=True)
    
    # Print statistics
    print("\n BASKET STATISTICS:")
    stats = builder.get_basket_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Print sample baskets
    print("\n SAMPLE BASKETS:")
    for i, basket in enumerate(baskets[:5], 1):
        print(f"   {i}. {basket}")
    
    # Print sample sequences
    print("\n SAMPLE SEQUENCES:")
    for i, sequence in enumerate(sequences[:5], 1):
        print(f"   {i}. {' → '.join(sequence)}")
    
    # Save preprocessed data
    builder.save()
    
    print("\n Preprocessing complete!")