"""
Decision Engine - Recommendation Orchestrator

Combines Association Rules + Sequential Patterns + (Future: Uplift Model)
to generate final recommendations
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_engine.association.fp_growth import AssociationEngine
from ml_engine.sequence.markov import MarkovSequenceModel


class RecommendationEngine:
    """
    Main recommendation orchestrator
    
    Combines multiple ML models to generate recommendations
    """
    
    def __init__(
        self,
        association_weight: float = 0.6,
        sequence_weight: float = 0.4,
        min_score_threshold: float = 0.15
    ):
        """
        Args:
            association_weight: Weight for association rule scores (0-1)
            sequence_weight: Weight for sequential pattern scores (0-1)
            min_score_threshold: Minimum combined score to recommend
        """
        self.association_weight = association_weight
        self.sequence_weight = sequence_weight
        self.min_score_threshold = min_score_threshold
        
        # Initialize model components
        self.association_engine = None
        self.sequence_model = None
        
        # Will be added in Week 4
        self.uplift_model = None
        
        print(f"🎯 Recommendation Engine initialized")
        print(f"   Association weight: {association_weight}")
        print(f"   Sequence weight: {sequence_weight}")
        print(f"   Min score threshold: {min_score_threshold}")
    
    def load_models(
        self,
        association_model_path: str = "models/association_rules.pkl",
        sequence_model_path: str = "models/sequence_model.pkl"
    ):
        """
        Load trained models
        """
        print("\n📂 Loading trained models...")
        
        # Load association rules
        self.association_engine = AssociationEngine()
        self.association_engine.load(association_model_path)
        
        # Load sequence model
        self.sequence_model = MarkovSequenceModel()
        self.sequence_model.load(sequence_model_path)
        
        print("✅ All models loaded")
    
    def get_candidates(
        self,
        current_basket: List[str],
        max_candidates: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Stage 1: Generate candidate items from both models
        
        Returns:
            Dictionary: {
                'item': {
                    'association_score': float,
                    'sequence_score': float,
                    'sources': list
                }
            }
        """
        candidates = {}
        
        # Get candidates from association rules
        if self.association_engine:
            assoc_recommendations = self.association_engine.get_recommendations(
                current_basket,
                max_items=max_candidates
            )
            
            for item, metrics in assoc_recommendations.items():
                if item not in candidates:
                    candidates[item] = {
                        'association_score': 0.0,
                        'sequence_score': 0.0,
                        'sources': []
                    }
                
                candidates[item]['association_score'] = metrics['confidence']
                candidates[item]['sources'].append('association')
                candidates[item]['association_metrics'] = metrics
        
        # Get candidates from sequential patterns
        if self.sequence_model:
            seq_predictions = self.sequence_model.predict_next(
                current_basket,
                max_predictions=max_candidates,
                strategy='last'  # Use last item
            )
            
            for item, prob in seq_predictions.items():
                if item not in candidates:
                    candidates[item] = {
                        'association_score': 0.0,
                        'sequence_score': 0.0,
                        'sources': []
                    }
                
                candidates[item]['sequence_score'] = prob
                candidates[item]['sources'].append('sequential')
        
        return candidates
    
    def score_candidates(
        self,
        candidates: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Stage 2: Calculate combined scores for candidates
        
        Combined Score = (w1 * association_score) + (w2 * sequence_score)
        
        Returns:
            Dictionary: {item: combined_score}
        """
        scored_candidates = {}
        
        for item, metrics in candidates.items():
            combined_score = (
                self.association_weight * metrics['association_score'] +
                self.sequence_weight * metrics['sequence_score']
            )
            
            scored_candidates[item] = combined_score
        
        return scored_candidates
    
    def rank_candidates(
        self,
        candidates: Dict[str, Dict[str, float]],
        scored_candidates: Dict[str, float],
        max_recommendations: int = 2
    ) -> List[Dict]:
        """
        Stage 3: Rank and filter candidates
        
        Returns top N candidates that meet minimum threshold
        """
        # Filter by threshold
        filtered = {
            item: score
            for item, score in scored_candidates.items()
            if score >= self.min_score_threshold
        }
        
        # Sort by score
        sorted_items = sorted(
            filtered.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build recommendation list with full details
        recommendations = []
        
        for item, combined_score in sorted_items[:max_recommendations]:
            rec = {
                'item': item,
                'combined_score': combined_score,
                'association_score': candidates[item]['association_score'],
                'sequence_score': candidates[item]['sequence_score'],
                'sources': candidates[item]['sources'],
                'reason': self._generate_reason(item, candidates[item])
            }
            
            # Add detailed metrics if available
            if 'association_metrics' in candidates[item]:
                rec['association_metrics'] = candidates[item]['association_metrics']
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_reason(self, item: str, metrics: Dict) -> str:
        """
        Generate human-readable reason for recommendation
        """
        reasons = []
        
        if 'association' in metrics['sources']:
            conf = metrics['association_score']
            reasons.append(f"frequently bought together ({conf:.0%} confidence)")
        
        if 'sequential' in metrics['sources']:
            prob = metrics['sequence_score']
            reasons.append(f"commonly purchased next ({prob:.0%} probability)")
        
        if len(reasons) == 0:
            return "recommended based on shopping patterns"
        elif len(reasons) == 1:
            return reasons[0]
        else:
            return " and ".join(reasons)
    
    def get_recommendations(
        self,
        current_basket: List[str],
        channel: str = "online",
        max_recommendations: int = 2,
        explain: bool = True
    ) -> Optional[Dict]:
        """
        Main entry point: Get recommendations for a basket
        
        Args:
            current_basket: List of items/categories in basket
            channel: 'online' or 'offline' (for future use)
            max_recommendations: Maximum number of recommendations
            explain: Include explanations in output
        
        Returns:
            Dictionary with recommendations and metadata, or None
        """
        if not current_basket:
            return None
        
        # Stage 1: Generate candidates
        candidates = self.get_candidates(current_basket, max_candidates=10)
        
        if not candidates:
            return None
        
        # Stage 2: Score candidates
        scored_candidates = self.score_candidates(candidates)
        
        # Stage 3: Rank and filter
        recommendations = self.rank_candidates(
            candidates,
            scored_candidates,
            max_recommendations=max_recommendations
        )
        
        if not recommendations:
            return None
        
        # Build response
        response = {
            'basket': current_basket,
            'channel': channel,
            'recommendations': recommendations,
            'count': len(recommendations),
            'model_version': 'v1.0_association_sequence'
        }
        
        if explain:
            response['metadata'] = {
                'total_candidates': len(candidates),
                'candidates_above_threshold': len(
                    [s for s in scored_candidates.values() if s >= self.min_score_threshold]
                ),
                'threshold': self.min_score_threshold,
                'weights': {
                    'association': self.association_weight,
                    'sequence': self.sequence_weight
                }
            }
        
        return response
    
    def batch_recommend(
        self,
        baskets: List[List[str]],
        channel: str = "online"
    ) -> List[Optional[Dict]]:
        """
        Get recommendations for multiple baskets
        """
        return [
            self.get_recommendations(basket, channel=channel)
            for basket in baskets
        ]
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        """
        info = {
            'engine_version': 'v1.0',
            'models_loaded': []
        }
        
        if self.association_engine:
            info['models_loaded'].append('association_rules')
            info['association_rules_count'] = len(self.association_engine.rules_df)
        
        if self.sequence_model:
            info['models_loaded'].append('sequence_model')
            stats = self.sequence_model.get_statistics()
            info['sequence_transitions'] = stats['total_valid_transitions']
        
        return info


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("🎯 TESTING RECOMMENDATION ENGINE")
    print("="*70)
    
    # Initialize engine
    engine = RecommendationEngine(
        association_weight=0.6,
        sequence_weight=0.4,
        min_score_threshold=0.15
    )
    
    # Load models
    engine.load_models()
    
    # Get model info
    print("\n📊 MODEL INFO:")
    info = engine.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test recommendations
    print("\n" + "="*70)
    print("🧪 TESTING RECOMMENDATIONS")
    print("="*70)
    
    test_baskets = [
        {
            'basket': ['Produce'],
            'name': 'Single item'
        },
        {
            'basket': ['Pantry', 'Produce'],
            'name': 'Common combination'
        },
        {
            'basket': ['Dairy', 'Bakery', 'Produce'],
            'name': 'Multiple items'
        },
        {
            'basket': ['Meat', 'Produce', 'Pantry'],
            'name': 'Complete shopping'
        }
    ]
    
    for test in test_baskets:
        basket = test['basket']
        name = test['name']
        
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"Basket: {basket}")
        print('-'*70)
        
        result = engine.get_recommendations(
            basket,
            channel='online',
            max_recommendations=2,
            explain=True
        )
        
        if result:
            print(f"✅ Got {result['count']} recommendation(s):")
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n{i}. {rec['item']}")
                print(f"   Combined Score: {rec['combined_score']:.3f}")
                print(f"   - Association: {rec['association_score']:.3f}")
                print(f"   - Sequential: {rec['sequence_score']:.3f}")
                print(f"   Sources: {', '.join(rec['sources'])}")
                print(f"   Reason: {rec['reason']}")
            
            if 'metadata' in result:
                print(f"\nMetadata:")
                print(f"   Total candidates: {result['metadata']['total_candidates']}")
                print(f"   Above threshold: {result['metadata']['candidates_above_threshold']}")
        else:
            print("❌ No recommendations")
    
    # Test batch recommendations
    print("\n" + "="*70)
    print("🧪 BATCH RECOMMENDATIONS")
    print("="*70)
    
    batch_baskets = [
        ['Produce'],
        ['Dairy', 'Bakery'],
        ['Pantry']
    ]
    
    results = engine.batch_recommend(batch_baskets, channel='offline')
    
    print(f"\nProcessed {len(results)} baskets:")
    for i, result in enumerate(results, 1):
        if result:
            items = [rec['item'] for rec in result['recommendations']]
            print(f"{i}. {result['basket']} → {items}")
        else:
            print(f"{i}. {batch_baskets[i-1]} → No recommendations")
    
    print("\n✅ Recommendation engine testing complete!")