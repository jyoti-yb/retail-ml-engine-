"""
Metrics and Evaluation Module

Calculates business metrics and ML performance metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


class RecommendationMetrics:
    """
    Calculate metrics for recommendation system evaluation
    """
    
    def __init__(self):
        self.recommendations_log = []
    
    def log_recommendation(
        self,
        basket: List[str],
        recommendations: List[str],
        accepted: List[str] = None,
        basket_value_before: float = 0,
        basket_value_after: float = 0
    ):
        """
        Log a recommendation event
        
        Args:
            basket: Original basket
            recommendations: Items recommended
            accepted: Items that were accepted (if known)
            basket_value_before: Basket value before recommendation
            basket_value_after: Basket value after recommendation
        """
        self.recommendations_log.append({
            'basket': basket,
            'basket_size_before': len(basket),
            'recommendations': recommendations,
            'num_recommendations': len(recommendations),
            'accepted': accepted if accepted else [],
            'num_accepted': len(accepted) if accepted else 0,
            'basket_value_before': basket_value_before,
            'basket_value_after': basket_value_after
        })
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate all metrics from logged recommendations
        """
        if not self.recommendations_log:
            return {'error': 'No recommendations logged'}
        
        df = pd.DataFrame(self.recommendations_log)
        
        # Acceptance metrics
        total_shown = df['num_recommendations'].sum()
        total_accepted = df['num_accepted'].sum()
        acceptance_rate = (total_accepted / total_shown * 100) if total_shown > 0 else 0
        
        # Basket size metrics
        avg_basket_size_before = df['basket_size_before'].mean()
        avg_basket_size_after = (df['basket_size_before'] + df['num_accepted']).mean()
        basket_size_lift = (
            (avg_basket_size_after - avg_basket_size_before) / avg_basket_size_before * 100
        )
        
        # Basket value metrics
        baskets_with_values = df[df['basket_value_before'] > 0]
        if len(baskets_with_values) > 0:
            avg_value_before = baskets_with_values['basket_value_before'].mean()
            avg_value_after = baskets_with_values['basket_value_after'].mean()
            value_lift = (avg_value_after - avg_value_before) / avg_value_before * 100
        else:
            avg_value_before = 0
            avg_value_after = 0
            value_lift = 0
        
        # Recommendation coverage
        total_transactions = len(df)
        transactions_with_recs = len(df[df['num_recommendations'] > 0])
        coverage = transactions_with_recs / total_transactions * 100
        
        metrics = {
            'total_transactions': total_transactions,
            'recommendations_shown': total_shown,
            'recommendations_accepted': total_accepted,
            'acceptance_rate_percent': acceptance_rate,
            'avg_basket_size_before': avg_basket_size_before,
            'avg_basket_size_after': avg_basket_size_after,
            'basket_size_lift_percent': basket_size_lift,
            'avg_basket_value_before': avg_value_before,
            'avg_basket_value_after': avg_value_after,
            'basket_value_lift_percent': value_lift,
            'recommendation_coverage_percent': coverage
        }
        
        return metrics
    
    def print_metrics(self):
        """
        Print metrics in readable format
        """
        metrics = self.calculate_metrics()
        
        if 'error' in metrics:
            print(f"⚠️  {metrics['error']}")
            return
        
        print("\n" + "="*70)
        print("📊 RECOMMENDATION SYSTEM METRICS")
        print("="*70)
        
        print(f"\n📈 COVERAGE:")
        print(f"   Total transactions: {metrics['total_transactions']}")
        print(f"   Recommendations shown: {metrics['recommendations_shown']}")
        print(f"   Coverage: {metrics['recommendation_coverage_percent']:.1f}%")
        
        print(f"\n✅ ACCEPTANCE:")
        print(f"   Recommendations accepted: {metrics['recommendations_accepted']}")
        print(f"   Acceptance rate: {metrics['acceptance_rate_percent']:.1f}%")
        
        print(f"\n🛒 BASKET SIZE:")
        print(f"   Before: {metrics['avg_basket_size_before']:.2f} items")
        print(f"   After: {metrics['avg_basket_size_after']:.2f} items")
        print(f"   Lift: {metrics['basket_size_lift_percent']:+.1f}%")
        
        if metrics['avg_basket_value_before'] > 0:
            print(f"\n💰 BASKET VALUE:")
            print(f"   Before: ₹{metrics['avg_basket_value_before']:.2f}")
            print(f"   After: ₹{metrics['avg_basket_value_after']:.2f}")
            print(f"   Lift: {metrics['basket_value_lift_percent']:+.1f}%")
        
        # Success criteria
        print(f"\n🎯 SUCCESS CRITERIA:")
        print(f"   Target basket size lift: 25%")
        print(f"   Current basket size lift: {metrics['basket_size_lift_percent']:.1f}%")
        
        if metrics['basket_size_lift_percent'] >= 25:
            print(f"   Status: ✅ TARGET ACHIEVED!")
        elif metrics['basket_size_lift_percent'] >= 20:
            print(f"   Status: 🟡 CLOSE TO TARGET")
        else:
            print(f"   Status: 🔴 BELOW TARGET")
    
    def clear_logs(self):
        """Clear logged recommendations"""
        self.recommendations_log = []


def simulate_recommendations_test(engine, test_baskets: List[Dict]) -> Dict:
    """
    Simulate recommendation system with test data
    
    Args:
        engine: RecommendationEngine instance
        test_baskets: List of test basket dictionaries with:
            - basket: List[str]
            - value: float (optional)
            - will_accept: bool (simulated acceptance)
    
    Returns:
        Dictionary of metrics
    """
    metrics = RecommendationMetrics()
    
    for test in test_baskets:
        basket = test['basket']
        basket_value = test.get('value', 0)
        will_accept = test.get('will_accept', False)
        
        # Get recommendations
        result = engine.get_recommendations(basket)
        
        if result and result['recommendations']:
            recommendations = [rec['item'] for rec in result['recommendations']]
            
            # Simulate acceptance
            accepted = []
            if will_accept and recommendations:
                # Accept first recommendation (simulated)
                accepted = [recommendations[0]]
                basket_value_after = basket_value + 50  # Simulate added value
            else:
                basket_value_after = basket_value
            
            metrics.log_recommendation(
                basket=basket,
                recommendations=recommendations,
                accepted=accepted,
                basket_value_before=basket_value,
                basket_value_after=basket_value_after
            )
        else:
            # No recommendations
            metrics.log_recommendation(
                basket=basket,
                recommendations=[],
                accepted=[],
                basket_value_before=basket_value,
                basket_value_after=basket_value
            )
    
    return metrics.calculate_metrics()


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Metrics Module")
    
    # Create sample data
    metrics = RecommendationMetrics()
    
    # Log some sample recommendations
    sample_logs = [
        {
            'basket': ['Produce', 'Pantry'],
            'recommendations': ['Dairy'],
            'accepted': ['Dairy'],
            'value_before': 150,
            'value_after': 200
        },
        {
            'basket': ['Dairy'],
            'recommendations': ['Bakery', 'Produce'],
            'accepted': ['Bakery'],
            'value_before': 100,
            'value_after': 135
        },
        {
            'basket': ['Pantry', 'Produce', 'Dairy'],
            'recommendations': ['Meat'],
            'accepted': [],
            'value_before': 250,
            'value_after': 250
        },
        {
            'basket': ['Bakery'],
            'recommendations': ['Dairy'],
            'accepted': ['Dairy'],
            'value_before': 80,
            'value_after': 140
        }
    ]
    
    for log in sample_logs:
        metrics.log_recommendation(
            basket=log['basket'],
            recommendations=log['recommendations'],
            accepted=log['accepted'],
            basket_value_before=log['value_before'],
            basket_value_after=log['value_after']
        )
    
    # Print metrics
    metrics.print_metrics()
    
    print("\n✅ Metrics module test complete!")