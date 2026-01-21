"""
Uplift Training Data Generator

Generates synthetic treatment/control data for uplift modeling.

This simulates what would happen if we ran an A/B test:
- Treatment group: Received recommendation
- Control group: No recommendation shown

Key insight: Some items have high natural purchase rate (would buy anyway),
while others have high uplift (only bought when suggested).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import pickle
from pathlib import Path
from typing import List, Dict, Tuple


class UpliftDataGenerator:
    """
    Generate synthetic uplift training data
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.transactions = []
        self.uplift_params = {}
        
    def define_uplift_profiles(self) -> Dict:
        """
        Define uplift profiles for different item combinations
        
        Returns:
            Dictionary with uplift parameters for each item pair
        """
        # Format: {(basket_item, candidate_item): (baseline_prob, uplift)}
        # baseline_prob: P(add | no prompt) - natural purchase rate
        # uplift: Additional probability when prompted
        
        profiles = {
            # HIGH UPLIFT items (big impact when suggested)
            ('Pantry', 'Produce'): {
                'baseline': 0.15,  # 15% buy anyway
                'uplift': 0.25,    # +25% when prompted → 40% total
                'reason': 'complementary_shopping'
            },
            ('Produce', 'Pantry'): {
                'baseline': 0.18,
                'uplift': 0.22,
                'reason': 'complementary_shopping'
            },
            ('Dairy', 'Bakery'): {
                'baseline': 0.12,
                'uplift': 0.28,
                'reason': 'breakfast_combo'
            },
            ('Bakery', 'Dairy'): {
                'baseline': 0.14,
                'uplift': 0.24,
                'reason': 'breakfast_combo'
            },
            
            # MEDIUM UPLIFT items
            ('Produce', 'Meat'): {
                'baseline': 0.20,
                'uplift': 0.15,
                'reason': 'meal_preparation'
            },
            ('Meat', 'Produce'): {
                'baseline': 0.22,
                'uplift': 0.13,
                'reason': 'meal_preparation'
            },
            ('Pantry', 'Dairy'): {
                'baseline': 0.16,
                'uplift': 0.12,
                'reason': 'staple_items'
            },
            
            # LOW UPLIFT items (high baseline - would buy anyway)
            ('Dairy', 'Produce'): {
                'baseline': 0.35,  # Already likely to buy
                'uplift': 0.05,    # Small impact from suggestion
                'reason': 'common_combination'
            },
            ('Produce', 'Dairy'): {
                'baseline': 0.32,
                'uplift': 0.06,
                'reason': 'common_combination'
            },
            
            # NEGATIVE UPLIFT items (reactance - prompting reduces purchase)
            ('Beverages', 'Meat'): {
                'baseline': 0.10,
                'uplift': -0.03,  # Suggesting feels pushy
                'reason': 'unrelated_reactance'
            },
            
            # Additional combinations
            ('Bakery', 'Pantry'): {'baseline': 0.14, 'uplift': 0.16, 'reason': 'baking'},
            ('Pantry', 'Meat'): {'baseline': 0.18, 'uplift': 0.10, 'reason': 'cooking'},
            ('Meat', 'Dairy'): {'baseline': 0.12, 'uplift': 0.14, 'reason': 'protein'},
            ('Dairy', 'Meat'): {'baseline': 0.11, 'uplift': 0.13, 'reason': 'protein'},
            ('Produce', 'Beverages'): {'baseline': 0.25, 'uplift': 0.08, 'reason': 'health'},
            ('Beverages', 'Produce'): {'baseline': 0.22, 'uplift': 0.09, 'reason': 'health'},
        }
        
        self.uplift_params = profiles
        return profiles
    
    def generate_transaction(
        self,
        transaction_id: int,
        basket: List[str],
        candidate_item: str,
        was_prompted: bool,
        timestamp: datetime,
        channel: str
    ) -> Dict:
        """
        Generate a single transaction with uplift simulation
        
        Args:
            transaction_id: Unique transaction ID
            basket: Current basket items
            candidate_item: Item being considered
            was_prompted: Whether user was shown recommendation
            timestamp: Transaction time
            channel: 'online' or 'offline'
        
        Returns:
            Dictionary with transaction data
        """
        # Get uplift parameters
        basket_key = basket[-1] if basket else 'unknown'  # Use last item
        param_key = (basket_key, candidate_item)
        
        if param_key in self.uplift_params:
            params = self.uplift_params[param_key]
            baseline_prob = params['baseline']
            uplift = params['uplift']
        else:
            # Default for unknown combinations
            baseline_prob = 0.10
            uplift = 0.05
        
        # Calculate purchase probability
        if was_prompted:
            # Treatment group: baseline + uplift
            purchase_prob = baseline_prob + uplift
        else:
            # Control group: baseline only
            purchase_prob = baseline_prob
        
        # Simulate purchase decision
        did_add = np.random.random() < purchase_prob
        
        # Calculate basket features
        basket_size = len(basket)
        basket_value = basket_size * 45  # Avg ₹45 per item
        
        # Time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Time of day categories
        if 6 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        # Customer segment (simulated)
        customer_segments = ['budget', 'regular', 'premium']
        customer_segment = random.choice(customer_segments)
        
        transaction = {
            'transaction_id': f"TXN_{transaction_id:06d}",
            'timestamp': timestamp,
            'channel': channel,
            'basket_size': basket_size,
            'basket_value': basket_value,
            'basket_items': ','.join(basket),
            'candidate_category': candidate_item,
            'was_prompted': 1 if was_prompted else 0,
            'did_add': 1 if did_add else 0,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'time_of_day': time_of_day,
            'is_online': 1 if channel == 'online' else 0,
            'customer_segment': customer_segment,
            'true_baseline_prob': baseline_prob,
            'true_uplift': uplift,
            'true_treatment_prob': baseline_prob + uplift
        }
        
        return transaction
    
    def generate_dataset(
        self,
        num_transactions: int = 5000,
        treatment_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate complete uplift training dataset
        
        Args:
            num_transactions: Number of transactions to generate
            treatment_ratio: Proportion in treatment group (default 50/50 split)
        
        Returns:
            DataFrame with uplift training data
        """
        print(f"\n🎲 Generating Uplift Training Dataset...")
        print(f"   Target transactions: {num_transactions}")
        print(f"   Treatment ratio: {treatment_ratio:.0%}")
        
        # Define uplift profiles
        self.define_uplift_profiles()
        
        # Load existing baskets for realistic basket compositions
        try:
            with open("data/processed_baskets.pkl", 'rb') as f:
                data = pickle.load(f)
            available_baskets = data['baskets']
            print(f"   Using {len(available_baskets)} real basket patterns")
        except:
            print("   ⚠️  Could not load real baskets, using synthetic")
            available_baskets = [
                ['Produce'], ['Dairy'], ['Pantry'], ['Bakery'],
                ['Produce', 'Pantry'], ['Dairy', 'Bakery'],
                ['Meat', 'Produce'], ['Pantry', 'Dairy']
            ]
        
        # Define candidate items (departments)
        candidate_items = ['Produce', 'Dairy', 'Pantry', 'Bakery', 'Meat', 'Beverages']
        
        transactions = []
        
        # Generate transactions
        start_date = datetime.now() - timedelta(days=180)  # 6 months of data
        
        for i in range(num_transactions):
            # Random timestamp
            days_offset = random.randint(0, 180)
            hour = random.choices(
                range(8, 23),  # Shopping hours 8am-11pm
                weights=[1, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2],  # Peak evening
                k=1
            )[0]
            timestamp = start_date + timedelta(days=days_offset, hours=hour)
            
            # Select basket
            basket = random.choice(available_baskets).copy()
            
            # Select candidate item (not in basket)
            available_candidates = [c for c in candidate_items if c not in basket]
            if not available_candidates:
                continue
            
            candidate = random.choice(available_candidates)
            
            # Assign to treatment or control
            was_prompted = random.random() < treatment_ratio
            
            # Channel
            channel = 'online' if random.random() < 0.7 else 'offline'
            
            # Generate transaction
            transaction = self.generate_transaction(
                transaction_id=i + 1,
                basket=basket,
                candidate_item=candidate,
                was_prompted=was_prompted,
                timestamp=timestamp,
                channel=channel
            )
            
            transactions.append(transaction)
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Print statistics
        self._print_statistics(df)
        
        self.transactions = transactions
        return df
    
    def _print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics"""
        print(f"\n✅ Generated {len(df)} transactions\n")
        
        print("📊 DATASET STATISTICS:")
        print(f"   Treatment group: {df['was_prompted'].sum()} ({df['was_prompted'].mean():.1%})")
        print(f"   Control group: {(~df['was_prompted'].astype(bool)).sum()} ({(~df['was_prompted'].astype(bool)).mean():.1%})")
        
        print(f"\n   Overall acceptance rate: {df['did_add'].mean():.1%}")
        
        treatment = df[df['was_prompted'] == 1]
        control = df[df['was_prompted'] == 0]
        
        treatment_accept = treatment['did_add'].mean()
        control_accept = control['did_add'].mean()
        
        print(f"   Treatment acceptance: {treatment_accept:.1%}")
        print(f"   Control acceptance: {control_accept:.1%}")
        print(f"   📈 Observed uplift: {(treatment_accept - control_accept):.1%}")
        
        print(f"\n   Channel distribution:")
        print(f"   - Online: {df['is_online'].mean():.1%}")
        print(f"   - Offline: {(1 - df['is_online'].mean()):.1%}")
        
        print(f"\n   Basket size:")
        print(f"   - Mean: {df['basket_size'].mean():.2f}")
        print(f"   - Median: {df['basket_size'].median():.0f}")
        
        print(f"\n   Top candidate items:")
        top_candidates = df['candidate_category'].value_counts().head(5)
        for item, count in top_candidates.items():
            print(f"   - {item}: {count} ({count/len(df)*100:.1f}%)")
    
    def save(self, output_path: str = "data/uplift_training_data.csv"):
        """
        Save dataset to CSV
        """
        if not self.transactions:
            raise ValueError("No data generated yet. Run generate_dataset() first.")
        
        df = pd.DataFrame(self.transactions)
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 Saved uplift training data to {output_path}")
        
        # Also save uplift parameters for reference
        params_path = output_path.replace('.csv', '_params.json')
        import json
        with open(params_path, 'w') as f:
            json.dump(self.uplift_params, f, indent=2)
        
        print(f"💾 Saved uplift parameters to {params_path}")
        
        return output_path
    
    def analyze_uplift_potential(self, df: pd.DataFrame):
        """
        Analyze which items have highest uplift potential
        """
        print("\n" + "="*70)
        print("📊 UPLIFT POTENTIAL ANALYSIS")
        print("="*70)
        
        # Group by candidate item
        results = []
        
        for candidate in df['candidate_category'].unique():
            candidate_df = df[df['candidate_category'] == candidate]
            
            treatment = candidate_df[candidate_df['was_prompted'] == 1]
            control = candidate_df[candidate_df['was_prompted'] == 0]
            
            if len(treatment) > 0 and len(control) > 0:
                treatment_rate = treatment['did_add'].mean()
                control_rate = control['did_add'].mean()
                uplift = treatment_rate - control_rate
                
                results.append({
                    'item': candidate,
                    'treatment_rate': treatment_rate,
                    'control_rate': control_rate,
                    'uplift': uplift,
                    'sample_size': len(candidate_df)
                })
        
        # Sort by uplift
        results_df = pd.DataFrame(results).sort_values('uplift', ascending=False)
        
        print("\n🏆 TOP UPLIFT ITEMS:\n")
        for i, row in results_df.head(10).iterrows():
            print(f"{row['item']:.<20} Uplift: {row['uplift']:>6.1%} "
                  f"(Treatment: {row['treatment_rate']:.1%}, Control: {row['control_rate']:.1%})")
        
        return results_df


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🎲 UPLIFT TRAINING DATA GENERATOR")
    print("="*70)
    
    # Initialize generator
    generator = UpliftDataGenerator(seed=42)
    
    # Generate dataset
    df = generator.generate_dataset(
        num_transactions=5000,
        treatment_ratio=0.5  # 50/50 split
    )
    
    # Analyze uplift potential
    uplift_analysis = generator.analyze_uplift_potential(df)
    
    # Save dataset
    output_path = generator.save()
    
    print("\n" + "="*70)
    print("✅ UPLIFT DATA GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output file: {output_path}")
    print(f"📊 Total rows: {len(df)}")
    print(f"📊 Columns: {len(df.columns)}")
    
    print(f"\n🔍 Sample data:")
    print(df.head(10).to_string())
    
    print("\n📋 Next steps:")
    print("   1. Review data: cat data/uplift_training_data.csv | head -20")
    print("   2. Review parameters: cat data/uplift_training_data_params.json")
    print("   3. Proceed to feature engineering")