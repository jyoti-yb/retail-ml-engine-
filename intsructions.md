# 📘 WEEK 1: Complete Implementation Instructions

### **STEP 0: Environment Setup**
```bash
# use venv with python 3.10.X verison
# Navigate to backend
cd backend

# Update requirements
# (Use the updated requirements.txt artifact)

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mlxtend; import pandas; import sklearn; print('✅ All dependencies installed')"
```

---

### **STEP 1: Create Folder Structure**

```bash
# Create ML engine folders
mkdir -p ml_engine/{preprocessing,association,sequence,uplift,decision_engine,utils}
mkdir -p data
mkdir -p models
mkdir -p notebooks
mkdir -p scripts

# Create __init__.py files
touch ml_engine/__init__.py
touch ml_engine/preprocessing/__init__.py
touch ml_engine/association/__init__.py
touch ml_engine/sequence/__init__.py
touch ml_engine/uplift/__init__.py
touch ml_engine/decision_engine/__init__.py
touch ml_engine/utils/__init__.py

# Verify structure
tree ml_engine  # or 'ls -R ml_engine' on systems without tree
```

---

### **STEP 2: Generate Sample Data**

```bash
# Create scripts directory
mkdir -p scripts

# Copy the download_kaggle_data.py code into:
# scripts/download_kaggle_data.py

# Run data generation
python scripts/download_kaggle_data.py

# When prompted, choose option 2 (Generate sample data)
```

**Expected output:**
``` Generated sample data:
   - Products: 21 items
   - Transactions: ~9000 order lines
   - Unique transactions: 3000
   - Files saved to: backend/data/
```

**Verify files created:**
```bash
ls -lh data/
# Should see:
# - products.csv
# - transactions.csv
```

---

### **STEP 3: Run Basket Preprocessing**

```bash
# Copy basket_builder.py code into:
# ml_engine/preprocessing/basket_builder.py

# Run preprocessing
python ml_engine/preprocessing/basket_builder.py
```

**Expected output:**
```
✅ Loaded XX order lines
   Unique transactions: 3000
   Unique products: 21

✅ Built XXXX baskets
   Avg basket size: ~4-6

✅ Built XXXX sequences
   Avg sequence length: ~4-6

💾 Saved preprocessed data to data/processed_baskets.pkl
```

**Verify:**
```bash
ls -lh data/processed_baskets.pkl
# Should exist and be ~100KB-1MB
```

---

### **STEP 4: Train Association Rules**

```bash
# Copy fp_growth.py code into:
# ml_engine/association/fp_growth.py

# Run training
python ml_engine/association/fp_growth.py
```

**Expected output:**
```
✅ Prepared XXXX transactions with X unique items

⚙️  Running FP-Growth algorithm...
✅ Found XX frequent itemsets

⚙️  Generating association rules...
✅ Generated XX association rules

🏆 TOP 10 ASSOCIATION RULES:
1. IF {Pantry} → THEN {Produce}
   Confidence: XX% | Support: XX% | Lift: X.XX
...

💾 Saved association rules to models/
```

**Verify files:**
```bash
ls -lh models/
# Should see:
# - association_rules.pkl
# - association_rules.json (human-readable)
```

---

### **STEP 5: Explore in Jupyter (Optional)**

```bash
# Copy notebook code into:
# notebooks/01_association_rules_exploration.ipynb

# Start Jupyter
jupyter notebook notebooks/

# Open and run the notebook
```

---

### **STEP 6: Run Comprehensive Tests**

```bash
# Copy test_week1.py into backend/

# Run all tests
python test_week1.py
```

**Expected output:**
```
🚀 WEEK 1: COMPREHENSIVE TESTING 🚀

TEST 1: DATA GENERATION
✅ TEST 1 PASSED

TEST 2: BASKET PREPROCESSING
✅ TEST 2 PASSED

TEST 3: ASSOCIATION RULE MINING
✅ TEST 3 PASSED

TEST 4: END-TO-END WORKFLOW
✅ TEST 4 PASSED

WEEK 1 TEST SUMMARY
Total: 4/4 tests passed

🎉 ALL TESTS PASSED! Week 1 complete.
```

---

## 🔍 Verification Checklist

After completing Week 1, you should have:

- [ ] `data/products.csv` - Product catalog
- [ ] `data/transactions.csv` - Transaction data
- [ ] `data/processed_baskets.pkl` - Preprocessed baskets
- [ ] `models/association_rules.pkl` - Trained FP-Growth model
- [ ] `models/association_rules.json` - Human-readable rules
- [ ] All tests passing

**File structure should look like:**
```
backend/
├── ml_engine/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── basket_builder.py
│   └── association/
│       ├── __init__.py
│       └── fp_growth.py
├── data/
│   ├── products.csv
│   ├── transactions.csv
│   └── processed_baskets.pkl
├── models/
│   ├── association_rules.pkl
│   └── association_rules.json
├── scripts/
│   └── download_kaggle_data.py
└── test_week1.py
```

---

##  Manual Testing

Test recommendations manually:

```python
from ml_engine.association.fp_growth import AssociationEngine

# Load trained model
engine = AssociationEngine()
engine.load("models/association_rules.pkl")

# Test with your basket
my_basket = ['Pantry', 'Produce']
recommendations = engine.get_recommendations(my_basket, max_items=3)

print(f"Basket: {my_basket}")
print(f"Recommendations: {recommendations}")
```

---

##  Expected Results

you should see:

1. **Association Rules:**
   - 30-100 rules generated (depends on data)
   - Top rules with 40-70% confidence
   - Clear patterns like "Pantry → Produce"

2. **Basket Statistics:**
   - Average basket size: 4-6 items
   - Average basket value: ₹200-400
   - 3000 unique transactions

3. **Recommendation Quality:**
   - Relevant suggestions based on basket
   - High-confidence recommendations (>30%)
   - Explainable with support/confidence metrics

---

## ⚠️ Troubleshooting

### Issue: mlxtend import error
```bash
pip install --upgrade mlxtend
```

### Issue: No association rules generated
- Lower min_support to 0.03 (3%)
- Lower min_confidence to 0.20 (20%)
- Check if baskets have variety (not all same items)

### Issue: File not found errors
```bash
# Make sure you're in backend/ directory
pwd  # Should show .../freshmart_q/backend

# Check file exists
ls data/transactions.csv
```

### Issue: Jupyter kernel issues
```bash
python -m ipykernel install --user --name=venv
# Then select 'venv' kernel in Jupyter
```

---

## 🎯 Success Criteria

You've successfully completed Week 1 if:

1. ✅ All 4 tests pass
2. ✅ Association rules JSON file is readable and makes sense
3. ✅ You can get recommendations for sample baskets
4. ✅ Rules have reasonable confidence (>30%) and lift (>1.0)

---


## 💡 Tips

1. **Keep models small:** Start with department-level, not item-level
2. **Inspect JSON:** Review `association_rules.json` to understand patterns
3. **Adjust thresholds:** If too many/few rules, adjust min_support/confidence
4. **Document findings:** Note top 5 rules for presentation

---

## 🆘 Getting Help

If stuck, check:
1. Is Python 3.8+ installed? `python --version`
2. Are all files in correct locations?
3. Did data generation complete successfully?
4. Run tests one by one to isolate issues

**Common fix:**
```bash
# Clean start
rm -rf data/ models/
python scripts/download_kaggle_data.py
python ml_engine/preprocessing/basket_builder.py
python ml_engine/association/fp_growth.py
python test_week1.py
```