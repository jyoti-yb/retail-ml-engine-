# 📘 WEEK 2: Complete Implementation Instructions

## 🎯 Goals
- ✅ Build sequential pattern model (Markov chains)
- ✅ Create decision engine orchestrator
- ✅ Combine association + sequence recommendations
- ✅ Build metrics evaluation module
- ✅ Test combined recommendations

---

## 📋 Prerequisites

Make sure Week 1 is complete:
```bash
ls models/association_rules.pkl  # Should exist
ls data/processed_baskets.pkl    # Should exist
```

---

## 🚀 Step-by-Step Instructions

### **STEP 1: Create Sequential Pattern Engine**

```bash
# Copy code to:
# ml_engine/sequence/markov.py

# Run training
python ml_engine/sequence/markov.py
```

**Expected output:**
```
✅ Loaded XXXX sequences

🤖 Training Markov Chain Sequential Model...
   Sequences: XXXX
   Min transition count: 3

📊 Training Statistics:
   Total transitions: XXXX
   Unique states: X
   States with outgoing transitions: X

🏆 TOP 10 SEQUENTIAL PATTERNS:
1. Pantry → Produce
   P(next | current): XX% | Count: XXX

💾 Saved sequence model to models/
```

**Verify files created:**
```bash
ls -lh models/
# Should see:
# - sequence_model.pkl
# - sequence_transitions.json
```

---

### **STEP 2: Create Decision Engine**

```bash
# Copy code to:
# ml_engine/decision_engine/recommender.py

# Run testing
python ml_engine/decision_engine/recommender.py
```

**Expected output:**
```
🎯 Recommendation Engine initialized
   Association weight: 0.6
   Sequence weight: 0.4

📂 Loading trained models...
✅ All models loaded

📊 MODEL INFO:
   engine_version: v1.0
   models_loaded: ['association_rules', 'sequence_model']

🧪 TESTING RECOMMENDATIONS
...
✅ Recommendation engine testing complete!
```

---

### **STEP 3: Create Metrics Module**

```bash
# Copy code to:
# ml_engine/utils/metrics.py

# Run testing
python ml_engine/utils/metrics.py
```

**Expected output:**
```
📊 RECOMMENDATION SYSTEM METRICS
...
✅ Metrics module test complete!
```

---

### **STEP 4: Run Comprehensive Tests**

```bash
# Copy test_week2.py to backend/

# Run all tests
python test_week2.py
```

**Expected output:**
```
🚀 WEEK 2: COMPREHENSIVE TESTING 🚀

TEST 1: SEQUENTIAL PATTERN MINING
✅ TEST 1 PASSED

TEST 2: DECISION ENGINE
✅ TEST 2 PASSED

TEST 3: COMBINED RECOMMENDATIONS
✅ TEST 3 PASSED

TEST 4: METRICS CALCULATION
✅ TEST 4 PASSED

TEST 5: END-TO-END WORKFLOW
✅ TEST 5 PASSED

WEEK 2 TEST SUMMARY
Total: 5/5 tests passed

🎉 ALL TESTS PASSED! Week 2 complete.
```

---

### **STEP 5: Explore in Jupyter (Optional)**

```bash
# Copy notebook to:
# notebooks/02_combined_recommendations.ipynb

# Start Jupyter
jupyter notebook notebooks/

# Open and run the notebook
```

---

## 🔍 Verification Checklist

After completing Week 2, you should have:

- [ ] `models/sequence_model.pkl` - Trained Markov model
- [ ] `models/sequence_transitions.json` - Human-readable transitions
- [ ] `ml_engine/decision_engine/recommender.py` - Decision orchestrator
- [ ] `ml_engine/utils/metrics.py` - Metrics calculator
- [ ] All tests passing (5/5)

**File structure:**
```
backend/
├── ml_engine/
│   ├── sequence/
│   │   ├── __init__.py
│   │   └── markov.py                  ✅ NEW
│   ├── decision_engine/
│   │   ├── __init__.py
│   │   └── recommender.py             ✅ NEW
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                 ✅ NEW
├── models/
│   ├── association_rules.pkl
│   ├── association_rules.json
│   ├── sequence_model.pkl             ✅ NEW
│   └── sequence_transitions.json      ✅ NEW
├── notebooks/
│   ├── 01_association_rules_exploration.ipynb
│   └── 02_combined_recommendations.ipynb  ✅ NEW
└── test_week2.py                      ✅ NEW
```

---

## 🧪 Manual Testing

Test combined recommendations:

```python
from ml_engine.decision_engine.recommender import RecommendationEngine

# Initialize and load
engine = RecommendationEngine(
    association_weight=0.6,
    sequence_weight=0.4,
    min_score_threshold=0.15
)
engine.load_models()

# Get recommendations
basket = ['Pantry', 'Produce']
result = engine.get_recommendations(basket, max_recommendations=2)

print(f"Basket: {basket}")
if result:
    for rec in result['recommendations']:
        print(f"  - {rec['item']}: {rec['combined_score']:.3f}")
        print(f"    Sources: {rec['sources']}")
        print(f"    Reason: {rec['reason']}")
```

---

## 📊 Expected Results

After Week 2, you should see:

1. **Sequential Patterns:**
   - 20-50 valid transitions
   - Clear patterns like "Produce → Pantry"
   - Transition probabilities 10-40%

2. **Combined Recommendations:**
   - Recommendations from both models
   - Combined scores that balance both signals
   - 60-80% recommendation coverage

3. **Metrics:**
   - Simulated acceptance rate: ~15-25%
   - Basket size lift: ~10-20% (without uplift yet)
   - Clear improvement over random

---

## 🎯 Key Differences from Week 1

| Aspect | Week 1 | Week 2 |
|--------|--------|--------|
| **Models** | Association only | Association + Sequence |
| **Decision** | Direct rules | Weighted combination |
| **Predictions** | Static correlations | Sequential patterns |
| **Coverage** | ~60% | ~70-80% |
| **Quality** | Good | Better (two signals) |

---

## ⚠️ Troubleshooting

### Issue: No transitions learned
```python
# Lower the min_transition_count
model = MarkovSequenceModel(min_transition_count=2)
```

### Issue: No recommendations from decision engine
```python
# Lower the score threshold
engine = RecommendationEngine(
    association_weight=0.6,
    sequence_weight=0.4,
    min_score_threshold=0.10  # Lower threshold
)
```

### Issue: Only one model contributing
- Check that both models are loaded
- Verify sequences exist in preprocessed data
- Test each model individually first

---

## 💡 Understanding the Flow

```
User adds "Pantry" to cart
        ↓
Decision Engine receives ['Pantry']
        ↓
Association Engine: "People who buy Pantry also buy..."
   → Produce (68% confidence)
   → Dairy (52% confidence)
        ↓
Sequence Model: "After Pantry, people usually buy..."
   → Produce (35% probability)
   → Meat (22% probability)
        ↓
Candidate Pool: {Produce, Dairy, Meat}
        ↓
Scoring:
   Produce: 0.6 * 0.68 + 0.4 * 0.35 = 0.548
   Dairy:   0.6 * 0.52 + 0.4 * 0.00 = 0.312
   Meat:    0.6 * 0.00 + 0.4 * 0.22 = 0.088
        ↓
Filter by threshold (0.15):
   ✅ Produce: 0.548
   ✅ Dairy:   0.312
   ❌ Meat:    0.088 (below threshold)
        ↓
Return top 2: [Produce, Dairy]
```

---

## 🎨 Visualization Tips

In the Jupyter notebook, you can visualize:

1. **Source Distribution:**
   - How many recs from association only
   - How many from sequence only
   - How many from both

2. **Score Distribution:**
   - Distribution of combined scores
   - Impact of different weight combinations

3. **Coverage Over Time:**
   - % of baskets receiving recommendations

---

## 🎯 Success Criteria

You've successfully completed Week 2 if:

1. ✅ All 5 tests pass
2. ✅ Sequential model finds meaningful patterns
3. ✅ Decision engine combines both models
4. ✅ Recommendations have diverse sources
5. ✅ Coverage improves over Week 1

---

## 📝 Next Steps

**Week 3 Preview:**
- Generate uplift training data
- Simulate treatment vs control groups
- Prepare features for causal modeling

**Week 4 Preview:**
- Build T-Learner uplift model
- Train on simulated data
- Integrate into decision engine

---

## 🎉 Achievements Unlocked

After Week 2, you can now:

1. ✅ Predict next likely purchases based on sequence
2. ✅ Combine multiple ML signals intelligently
3. ✅ Explain recommendations with multiple reasons
4. ✅ Measure recommendation performance
5. ✅ Tune model weights for optimal results

---

## 🆘 Need Help?

Common issues:

1. **Models not loading:** Check paths in load_models()
2. **Low coverage:** Lower min_score_threshold
3. **Imbalanced sources:** Adjust weights
4. **Poor metrics:** Need uplift model (Week 4)

---

**Ready for Week 3?** Once you confirm Week 2 is working, I'll give you the uplift data generation code!