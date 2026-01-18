"""
Download and prepare Kaggle grocery dataset for ML engine

Dataset: Instacart Market Basket Analysis
URL: https://www.kaggle.com/c/instacart-market-basket-analysis
(or use smaller alternatives listed below)
"""

import pandas as pd
import os
from pathlib import Path

def download_data():
    """
    Download grocery dataset from Kaggle
    
    OPTION 1: If you have Kaggle API configured
    Run: kaggle competitions download -c instacart-market-basket-analysis
    
    OPTION 2: Manual download
    1. Go to: https://www.kaggle.com/c/instacart-market-basket-analysis/data
    2. Download 'orders.csv' and 'order_products__train.csv'
    3. Place in backend/data/
    
    OPTION 3: Use alternative smaller dataset (RECOMMENDED FOR TESTING)
    """
    
    print("📥 Preparing Kaggle Grocery Dataset...")
    print("\nOPTIONS:")
    print("1. Instacart (large, real)")
    print("2. Generate sample data (for quick testing)")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "2":
        generate_sample_data()
    else:
        print("\n⚠️  For Instacart dataset:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("3. Run: kaggle competitions download -c instacart-market-basket-analysis")
        print("4. Extract to backend/data/")
        print("\nOr manually download from Kaggle website")

def generate_sample_data():
    """
    Generate realistic sample grocery data for testing
    """
    import random
    import datetime
    
    print("\n🔧 Generating sample grocery dataset...")
    
    # Define product catalog
    products = [
        # Produce
        ("Organic Bananas", "Produce", "Fruit", 0.69),
        ("Organic Strawberries", "Produce", "Fruit", 4.99),
        ("Organic Baby Spinach", "Produce", "Vegetables", 3.99),
        ("Organic Avocado", "Produce", "Fruit", 1.99),
        ("Organic Tomatoes", "Produce", "Vegetables", 2.49),
        
        # Dairy
        ("Whole Milk", "Dairy", "Milk", 3.99),
        ("Butter", "Dairy", "Butter", 4.49),
        ("Greek Yogurt", "Dairy", "Yogurt", 5.99),
        ("Cheddar Cheese", "Dairy", "Cheese", 6.99),
        
        # Grains
        ("Basmati Rice", "Pantry", "Rice", 8.99),
        ("Whole Wheat Bread", "Bakery", "Bread", 3.49),
        ("Pasta", "Pantry", "Pasta", 2.99),
        
        # Proteins
        ("Eggs", "Dairy", "Eggs", 4.99),
        ("Chicken Breast", "Meat", "Poultry", 8.99),
        ("Toor Dal", "Pantry", "Pulses", 4.99),
        
        # Pantry
        ("Olive Oil", "Pantry", "Oils", 9.99),
        ("Sugar", "Pantry", "Baking", 3.49),
        ("Salt", "Pantry", "Spices", 1.99),
        ("Black Pepper", "Pantry", "Spices", 4.99),
        
        # Beverages
        ("Orange Juice", "Beverages", "Juice", 4.99),
        ("Coffee", "Beverages", "Coffee", 12.99),
    ]
    
    # Create product dataframe
    products_df = pd.DataFrame(products, columns=['product_name', 'department', 'aisle', 'price'])
    products_df['product_id'] = range(1, len(products) + 1)
    
    # Generate transactions
    transactions = []
    transaction_id = 1
    
    # Common shopping patterns (for realistic associations)
    common_patterns = [
        ["Basmati Rice", "Toor Dal", "Olive Oil"],  # Cooking from scratch
        ["Whole Milk", "Eggs", "Whole Wheat Bread"],  # Breakfast
        ["Organic Bananas", "Organic Strawberries", "Greek Yogurt"],  # Healthy snack
        ["Chicken Breast", "Organic Tomatoes", "Organic Baby Spinach"],  # Dinner
        ["Pasta", "Organic Tomatoes", "Olive Oil"],  # Italian
        ["Coffee", "Whole Milk", "Sugar"],  # Morning routine
    ]
    
    # Generate 3000 transactions
    num_transactions = 3000
    
    for i in range(num_transactions):
        # Randomly choose: pattern-based (60%) or random (40%)
        if random.random() < 0.6 and common_patterns:
            # Use a common pattern
            pattern = random.choice(common_patterns)
            basket_items = pattern.copy()
            
            # Add 1-3 random items
            num_extra = random.randint(1, 3)
            extra_items = random.sample([p[0] for p in products], num_extra)
            basket_items.extend(extra_items)
        else:
            # Completely random basket
            basket_size = random.randint(3, 12)
            basket_items = random.sample([p[0] for p in products], basket_size)
        
        # Remove duplicates
        basket_items = list(set(basket_items))
        
        # Create transaction timestamp (random over last 6 months)
        days_ago = random.randint(0, 180)
        hours = random.randint(8, 22)  # Shopping hours 8am-10pm
        timestamp = datetime.datetime.now() - datetime.timedelta(days=days_ago, hours=hours)
        
        # Channel: 70% online, 30% offline
        channel = "online" if random.random() < 0.7 else "offline"
        
        # Add to transactions list
        for item_name in basket_items:
            product = products_df[products_df['product_name'] == item_name].iloc[0]
            transactions.append({
                'transaction_id': f"TXN_{transaction_id:05d}",
                'customer_id': f"CUST_{random.randint(1, 500):04d}",  # 500 unique customers
                'timestamp': timestamp,
                'channel': channel,
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'department': product['department'],
                'aisle': product['aisle'],
                'price': product['price']
            })
        
        transaction_id += 1
    
    transactions_df = pd.DataFrame(transactions)
    
    # Save to CSV
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    products_df.to_csv(data_dir / "products.csv", index=False)
    transactions_df.to_csv(data_dir / "transactions.csv", index=False)
    
    print(f"\n✅ Generated sample data:")
    print(f"   - Products: {len(products_df)} items")
    print(f"   - Transactions: {len(transactions_df)} order lines")
    print(f"   - Unique transactions: {transaction_id - 1}")
    print(f"   - Files saved to: backend/data/")
    print(f"\n📊 Sample transaction:")
    print(transactions_df.head(10))
    
    return products_df, transactions_df

if __name__ == "__main__":
    download_data()