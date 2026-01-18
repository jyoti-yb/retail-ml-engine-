"""
Association Rule Mining using FP-Growth Algorithm

Discovers which items are frequently bought together
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle
import json
from pathlib import Path


class AssociationEngine:
    """
    FP-Growth based association rule mining
    """
    
    def __init__(
        self, 
        min_support: float = 0.05,
        min_confidence: float = 0.30,
        min_lift: float = 1.0
    ):
        """
        Args:
            min_support: Minimum support threshold (default 5%)
            min_confidence: Minimum confidence threshold (default 30%)
            min_lift: Minimum lift threshold (default 1.0)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        
        self.rules_df = None
        self.frequent_itemsets = None
        self.rules_dict = {}  # For fast lookup: {antecedent: [consequents]}
    
    def prepare_data(self, baskets: List[List[str]]) -> pd.DataFrame:
        """
        Convert basket list to one-hot encoded DataFrame
        
        Args:
            baskets: List of baskets, e.g., [['Rice', 'Dal'], ['Bread', 'Butter']]
        
        Returns:
            One-hot encoded DataFrame suitable for FP-Growth
        """
        print("🔧 Preparing data for FP-Growth...")
        
        # Use TransactionEncoder for one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"✅ Prepared {len(df)} transactions with {len(df.columns)} unique items")
        
        return df
    
    def train(self, baskets: List[List[str]]):
        """
        Train FP-Growth and generate association rules
        
        Args:
            baskets: List of baskets
        """
        print(f"\n🤖 Training FP-Growth Association Rules...")
        print(f"   Min Support: {self.min_support}")
        print(f"   Min Confidence: {self.min_confidence}")
        print(f"   Min Lift: {self.min_lift}")
        
        # Prepare data
        df = self.prepare_data(baskets)
        
        # Run FP-Growth to find frequent itemsets
        print("\n⚙️  Running FP-Growth algorithm...")
        self.frequent_itemsets = fpgrowth(
            df,
            min_support=self.min_support,
            use_colnames=True
        )
        
        print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        print("\n⚙️  Generating association rules...")
        self.rules_df = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        
        # Filter by lift
        self.rules_df = self.rules_df[self.rules_df['lift'] >= self.min_lift]
        
        # Sort by confidence (descending)
        self.rules_df = self.rules_df.sort_values('confidence', ascending=False)
        
        print(f"✅ Generated {len(self.rules_df)} association rules")
        
        # Build fast lookup dictionary
        self._build_lookup_dict()
        
        # Print top rules
        self._print_top_rules(n=10)
    
    def _build_lookup_dict(self):
        """
        Build dictionary for fast rule lookup
        Format: {antecedent_item: [(consequent, confidence, lift), ...]}
        """
        print("\n🗂️  Building rule lookup dictionary...")
        
        self.rules_dict = {}
        
        for _, row in self.rules_df.iterrows():
            # Convert frozensets to lists
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            # For single-item antecedents (most common case)
            if len(antecedents) == 1:
                antecedent = antecedents[0]
                
                if antecedent not in self.rules_dict:
                    self.rules_dict[antecedent] = []
                
                # Store consequent with metrics
                for consequent in consequents:
                    self.rules_dict[antecedent].append({
                        'item': consequent,
                        'confidence': row['confidence'],
                        'support': row['support'],
                        'lift': row['lift']
                    })
        
        print(f"✅ Built lookup dict with {len(self.rules_dict)} antecedent items")
    
    def _print_top_rules(self, n: int = 10):
        """Print top N rules"""
        print(f"\n🏆 TOP {n} ASSOCIATION RULES:\n")
        
        for i, (_, row) in enumerate(self.rules_df.head(n).iterrows(), 1):
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            
            print(f"{i}. IF {{{antecedents}}} → THEN {{{consequents}}}")
            print(f"   Confidence: {row['confidence']:.2%} | "
                  f"Support: {row['support']:.2%} | "
                  f"Lift: {row['lift']:.2f}\n")
    
    def get_recommendations(
        self, 
        current_basket: List[str],
        max_items: int = 5
    ) -> Dict[str, Dict]:
        """
        Get item recommendations based on current basket
        
        Args:
            current_basket: List of items currently in basket
            max_items: Maximum number of recommendations
        
        Returns:
            Dictionary: {item: {confidence, support, lift}}
        """
        recommendations = {}
        
        # For each item in current basket
        for item in current_basket:
            if item in self.rules_dict:
                # Get all consequents for this item
                for rule in self.rules_dict[item]:
                    consequent = rule['item']
                    
                    # Don't recommend items already in basket
                    if consequent not in current_basket:
                        # Keep highest confidence if item appears multiple times
                        if consequent not in recommendations or \
                           rule['confidence'] > recommendations[consequent]['confidence']:
                            recommendations[consequent] = {
                                'confidence': rule['confidence'],
                                'support': rule['support'],
                                'lift': rule['lift'],
                                'source_item': item
                            }
        
        # Sort by confidence and return top N
        sorted_recs = dict(
            sorted(
                recommendations.items(),
                key=lambda x: x[1]['confidence'],
                reverse=True
            )[:max_items]
        )
        
        return sorted_recs
    
    def get_rule_explanation(self, antecedent: str, consequent: str) -> str:
        """
        Get human-readable explanation for a rule
        """
        if antecedent in self.rules_dict:
            for rule in self.rules_dict[antecedent]:
                if rule['item'] == consequent:
                    confidence = rule['confidence']
                    support = rule['support']
                    return (
                        f"{confidence:.0%} of customers who buy {antecedent} "
                        f"also buy {consequent} "
                        f"(appears in {support:.0%} of all transactions)"
                    )
        
        return "No rule found for this combination"
    
    def save(self, output_dir: str = "models"):
        """
        Save trained model
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as pickle
        model_data = {
            'rules_df': self.rules_df,
            'frequent_itemsets': self.frequent_itemsets,
            'rules_dict': self.rules_dict,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'min_lift': self.min_lift
        }
        
        with open(output_path / "association_rules.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save rules as JSON for easy inspection
        rules_json = []
        for _, row in self.rules_df.iterrows():
            rules_json.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'confidence': float(row['confidence']),
                'support': float(row['support']),
                'lift': float(row['lift'])
            })
        
        with open(output_path / "association_rules.json", 'w') as f:
            json.dump(rules_json, f, indent=2)
        
        print(f"\n💾 Saved association rules to {output_dir}/")
        print(f"   - association_rules.pkl (model)")
        print(f"   - association_rules.json (human-readable)")
    
    def load(self, model_path: str = "models/association_rules.pkl"):
        """
        Load trained model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rules_df = model_data['rules_df']
        self.frequent_itemsets = model_data['frequent_itemsets']
        self.rules_dict = model_data['rules_dict']
        self.min_support = model_data['min_support']
        self.min_confidence = model_data['min_confidence']
        self.min_lift = model_data['min_lift']
        
        print(f"✅ Loaded association rules from {model_path}")
        print(f"   Rules: {len(self.rules_df)}")
        print(f"   Frequent itemsets: {len(self.frequent_itemsets)}")


# Example usage and testing
if __name__ == "__main__":
    # Load preprocessed baskets
    print("📂 Loading preprocessed baskets...")
    with open("data/processed_baskets.pkl", 'rb') as f:
        data = pickle.load(f)
    
    baskets = data['baskets']
    print(f"✅ Loaded {len(baskets)} baskets")
    
    # Initialize and train
    engine = AssociationEngine(
        min_support=0.05,     # 5% of transactions
        min_confidence=0.30,  # 30% confidence
        min_lift=1.0          # Lift > 1.0
    )
    
    engine.train(baskets)
    
    # Test recommendations
    print("\n" + "="*60)
    print(" TESTING RECOMMENDATIONS")
    print("="*60)
    
    test_baskets = [
        ['Pantry', 'Produce'],
        ['Dairy', 'Bakery'],
        ['Meat', 'Produce']
    ]
    
    for basket in test_baskets:
        print(f"\n📦 Current Basket: {basket}")
        recommendations = engine.get_recommendations(basket, max_items=3)
        
        if recommendations:
            print("   Recommendations:")
            for item, metrics in recommendations.items():
                print(f"   - {item}: {metrics['confidence']:.1%} confidence")
                explanation = engine.get_rule_explanation(
                    metrics['source_item'], 
                    item
                )
                print(f"     {explanation}")
        else:
            print("   No recommendations found")
    
    # Save model
    engine.save()
    
    print("\n Association rule engine training complete!")