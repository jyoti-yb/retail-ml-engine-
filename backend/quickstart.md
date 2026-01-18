cd backend

# 1. Install dependencies
pip install -r requirements.txt

# 2. Create folders
mkdir -p ml_engine/{preprocessing,association,sequence,uplift,decision_engine,utils}
mkdir -p data models notebooks scripts

# 3. Generate data
python scripts/download_kaggle_data.py
# Choose option 2

# 4. Preprocess baskets
python ml_engine/preprocessing/basket_builder.py

# 5. Train association rules
python ml_engine/association/fp_growth.py

# 6. Run tests
python test_week1.py
```

---

## 📂 **File Placement Guide:**

Copy the artifacts I created to these locations:
```
backend/
├── requirements.txt              ← Artifact #1
├── scripts/
│   └── download_kaggle_data.py   ← Artifact #2
├── ml_engine/
│   ├── preprocessing/
│   │   └── basket_builder.py     ← Artifact #3
│   └── association/
│       └── fp_growth.py          ← Artifact #4
├── notebooks/
│   └── 01_association_rules_exploration.ipynb  ← Artifact #5
├── test_week1.py                 ← Artifact #6
├── WEEK1_INSTRUCTIONS.md         ← Artifact #7
└── run_week1.sh                  ← Artifact #8
```

---

## 🎯 **Expected Results:**

After running everything, you'll have:
```
✅ 3000 grocery transactions generated
✅ ~2500 baskets preprocessed
✅ 30-100 association rules discovered
✅ Working recommendation engine
✅ All tests passing (4/4)
```

**Example output:**
```
🏆 TOP ASSOCIATION RULES:
1. IF {Pantry} → THEN {Produce}
   Confidence: 65% | Support: 28% | Lift: 1.8
   
2. IF {Dairy} → THEN {Bakery}
   Confidence: 58% | Support: 22% | Lift: 1.6