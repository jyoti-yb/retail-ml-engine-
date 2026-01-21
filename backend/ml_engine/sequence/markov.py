"""
Sequential Pattern Mining using Markov Chains

Predicts next likely items based on purchase order/sequence
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import pickle
import json
from pathlib import Path


class MarkovSequenceModel:
    """
    First-order Markov Chain for sequence prediction
    
    Models transition probabilities: P(next_item | current_item)
    """
    
    def __init__(self, min_transition_count: int = 3):
        """
        Args:
            min_transition_count: Minimum times a transition must occur to be included
        """
        self.min_transition_count = min_transition_count
        
        # Raw transition counts
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
        # Normalized transition probabilities
        self.transition_probs = {}
        
        # State statistics
        self.state_counts = defaultdict(int)
        self.unique_states = set()
    
    def train(self, sequences: List[List[str]]):
        """
        Train Markov model on sequential data
        
        Args:
            sequences: List of ordered sequences
                      e.g., [['Vegetables', 'Rice', 'Dal'], ['Bread', 'Butter'], ...]
        """
        print(f"\n🤖 Training Markov Chain Sequential Model...")
        print(f"   Sequences: {len(sequences)}")
        print(f"   Min transition count: {self.min_transition_count}")
        
        # Count transitions
        total_transitions = 0
        
        for sequence in sequences:
            if len(sequence) < 2:
                continue  # Need at least 2 items for a transition
            
            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                
                # Record transition
                self.transition_counts[current_state][next_state] += 1
                self.state_counts[current_state] += 1
                
                self.unique_states.add(current_state)
                self.unique_states.add(next_state)
                
                total_transitions += 1
        
        print(f"\n📊 Training Statistics:")
        print(f"   Total transitions: {total_transitions}")
        print(f"   Unique states: {len(self.unique_states)}")
        print(f"   States with outgoing transitions: {len(self.transition_counts)}")
        
        # Normalize to probabilities
        self._normalize_transitions()
        
        # Print top transitions
        self._print_top_transitions(n=10)
    
    def _normalize_transitions(self):
        """
        Convert counts to probabilities
        P(next | current) = count(current → next) / count(current)
        """
        print("\n⚙️  Normalizing transition probabilities...")
        
        self.transition_probs = {}
        
        for current_state, next_states in self.transition_counts.items():
            total_count = self.state_counts[current_state]
            
            self.transition_probs[current_state] = {}
            
            for next_state, count in next_states.items():
                # Only include transitions that meet minimum count
                if count >= self.min_transition_count:
                    probability = count / total_count
                    self.transition_probs[current_state][next_state] = probability
        
        # Count valid transitions
        total_valid_transitions = sum(
            len(next_states) 
            for next_states in self.transition_probs.values()
        )
        
        print(f"✅ Valid transitions (count >= {self.min_transition_count}): {total_valid_transitions}")
    
    def _print_top_transitions(self, n: int = 10):
        """Print top N transitions by probability"""
        print(f"\n🏆 TOP {n} SEQUENTIAL PATTERNS:\n")
        
        # Collect all transitions with probabilities
        all_transitions = []
        for current_state, next_states in self.transition_probs.items():
            for next_state, prob in next_states.items():
                all_transitions.append((current_state, next_state, prob))
        
        # Sort by probability
        all_transitions.sort(key=lambda x: x[2], reverse=True)
        
        for i, (current, next_item, prob) in enumerate(all_transitions[:n], 1):
            count = self.transition_counts[current][next_item]
            print(f"{i}. {current} → {next_item}")
            print(f"   P(next | current): {prob:.2%} | Count: {count}\n")
    
    def predict_next(
        self, 
        current_basket: List[str],
        max_predictions: int = 5,
        strategy: str = 'last'
    ) -> Dict[str, float]:
        """
        Predict next likely items based on current basket
        
        Args:
            current_basket: List of items currently in basket
            max_predictions: Maximum number of predictions to return
            strategy: 'last' (use last item) or 'all' (aggregate from all items)
        
        Returns:
            Dictionary: {item: probability}
        """
        if not current_basket:
            return {}
        
        predictions = {}
        
        if strategy == 'last':
            # Use only the last item in basket
            last_item = current_basket[-1]
            
            if last_item in self.transition_probs:
                predictions = self.transition_probs[last_item].copy()
        
        elif strategy == 'all':
            # Aggregate predictions from all items in basket
            for item in current_basket:
                if item in self.transition_probs:
                    for next_item, prob in self.transition_probs[item].items():
                        # Take maximum probability if item appears from multiple sources
                        predictions[next_item] = max(
                            predictions.get(next_item, 0),
                            prob
                        )
        
        # Remove items already in basket
        predictions = {
            item: prob 
            for item, prob in predictions.items() 
            if item not in current_basket
        }
        
        # Sort and return top N
        sorted_predictions = dict(
            sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_predictions]
        )
        
        return sorted_predictions
    
    def get_sequence_probability(self, sequence: List[str]) -> float:
        """
        Calculate probability of a sequence occurring
        P(seq) = P(s1 → s2) * P(s2 → s3) * ...
        """
        if len(sequence) < 2:
            return 0.0
        
        prob = 1.0
        
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_item = sequence[i + 1]
            
            if current in self.transition_probs and \
               next_item in self.transition_probs[current]:
                prob *= self.transition_probs[current][next_item]
            else:
                return 0.0  # Transition doesn't exist
        
        return prob
    
    def get_explanation(self, current_item: str, next_item: str) -> str:
        """
        Get human-readable explanation for a transition
        """
        if current_item in self.transition_probs and \
           next_item in self.transition_probs[current_item]:
            prob = self.transition_probs[current_item][next_item]
            count = self.transition_counts[current_item][next_item]
            
            return (
                f"{prob:.0%} of customers who buy {current_item} "
                f"proceed to buy {next_item} next "
                f"(observed {count} times)"
            )
        
        return "No sequential pattern found for this transition"
    
    def get_statistics(self) -> Dict:
        """
        Get model statistics
        """
        stats = {
            'unique_states': len(self.unique_states),
            'states_with_transitions': len(self.transition_probs),
            'total_valid_transitions': sum(
                len(next_states) 
                for next_states in self.transition_probs.values()
            ),
            'avg_transitions_per_state': 0,
            'max_transitions_from_state': 0,
            'most_common_next_state': None
        }
        
        if stats['states_with_transitions'] > 0:
            stats['avg_transitions_per_state'] = (
                stats['total_valid_transitions'] / stats['states_with_transitions']
            )
        
        # Find state with most outgoing transitions
        if self.transition_probs:
            max_state = max(
                self.transition_probs.items(),
                key=lambda x: len(x[1])
            )
            stats['max_transitions_from_state'] = len(max_state[1])
            stats['state_with_most_transitions'] = max_state[0]
        
        return stats
    
    def save(self, output_dir: str = "models"):
        """
        Save trained model
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as pickle
        model_data = {
            'transition_counts': dict(self.transition_counts),
            'transition_probs': self.transition_probs,
            'state_counts': dict(self.state_counts),
            'unique_states': list(self.unique_states),
            'min_transition_count': self.min_transition_count
        }
        
        with open(output_path / "sequence_model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save transition matrix as JSON
        transitions_json = []
        for current, next_states in self.transition_probs.items():
            for next_item, prob in next_states.items():
                transitions_json.append({
                    'from': current,
                    'to': next_item,
                    'probability': float(prob),
                    'count': int(self.transition_counts[current][next_item])
                })
        
        # Sort by probability
        transitions_json.sort(key=lambda x: x['probability'], reverse=True)
        
        with open(output_path / "sequence_transitions.json", 'w') as f:
            json.dump(transitions_json, f, indent=2)
        
        print(f"\n💾 Saved sequence model to {output_dir}/")
        print(f"   - sequence_model.pkl (model)")
        print(f"   - sequence_transitions.json (human-readable)")
    
    def load(self, model_path: str = "models/sequence_model.pkl"):
        """
        Load trained model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Convert back to defaultdicts
        self.transition_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in model_data['transition_counts'].items()}
        )
        self.transition_probs = model_data['transition_probs']
        self.state_counts = defaultdict(int, model_data['state_counts'])
        self.unique_states = set(model_data['unique_states'])
        self.min_transition_count = model_data['min_transition_count']
        
        print(f"✅ Loaded sequence model from {model_path}")
        print(f"   States: {len(self.unique_states)}")
        print(f"   Valid transitions: {sum(len(n) for n in self.transition_probs.values())}")


# Example usage and testing
if __name__ == "__main__":
    # Load preprocessed sequences
    print("📂 Loading preprocessed sequences...")
    with open("data/processed_baskets.pkl", 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    print(f"✅ Loaded {len(sequences)} sequences")
    
    # Print sample sequences
    print("\n📊 SAMPLE SEQUENCES:")
    for i, seq in enumerate(sequences[:5], 1):
        print(f"   {i}. {' → '.join(seq)}")
    
    # Initialize and train
    model = MarkovSequenceModel(min_transition_count=3)
    model.train(sequences)
    
    # Get statistics
    print("\n" + "="*60)
    print("📊 MODEL STATISTICS")
    print("="*60)
    stats = model.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test predictions
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTIONS")
    print("="*60)
    
    test_baskets = [
        ['Produce'],
        ['Dairy', 'Bakery'],
        ['Pantry', 'Produce'],
        ['Meat', 'Produce', 'Pantry']
    ]
    
    for basket in test_baskets:
        print(f"\n📦 Current Basket: {basket}")
        
        # Last item strategy
        predictions_last = model.predict_next(basket, max_predictions=3, strategy='last')
        print(f"   Next likely (based on last item):")
        for item, prob in predictions_last.items():
            print(f"   - {item}: {prob:.1%}")
        
        # All items strategy
        predictions_all = model.predict_next(basket, max_predictions=3, strategy='all')
        print(f"   Next likely (based on all items):")
        for item, prob in predictions_all.items():
            print(f"   - {item}: {prob:.1%}")
    
    # Test sequence probability
    print("\n" + "="*60)
    print("🧪 SEQUENCE PROBABILITY")
    print("="*60)
    
    test_sequence = ['Produce', 'Pantry', 'Dairy']
    prob = model.get_sequence_probability(test_sequence)
    print(f"\nSequence: {' → '.join(test_sequence)}")
    print(f"P(sequence): {prob:.2%}")
    
    # Save model
    model.save()
    
    print("\n✅ Sequential pattern model training complete!")