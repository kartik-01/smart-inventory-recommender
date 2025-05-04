import os
import pandas as pd
from collections import Counter
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))


def load_edges(edges_filename='edges.csv'):
    """
    Load the co-purchase edges CSV from the data directory.
    """
    path = os.path.join(DATA_DIR, edges_filename)
    return pd.read_csv(path)


def build_transactions(edges_df):
    """
    Group edges by source ASIN to create co-purchase baskets.
    Returns a list of transaction lists.
    """
    return edges_df.groupby('source')['target'].apply(list).tolist()


def filter_top_items(transactions, top_n=500):
    """
    Reduce transactions to only the top_n most frequent items to limit dimensionality.
    """
    item_counts = Counter(item for basket in transactions for item in basket)
    top_items = set([item for item, _ in item_counts.most_common(top_n)])
    filtered = []
    for basket in transactions:
        fb = [item for item in basket if item in top_items]
        if len(fb) >= 2:
            filtered.append(fb)
    return filtered, top_items


def transactions_to_onehot(transactions, items):
    """
    Convert list of filtered transactions into a one-hot encoded DataFrame for given items.
    """
    onehot_rows = []
    for basket in transactions:
        row = {item: (item in basket) for item in items}
        onehot_rows.append(row)
    return pd.DataFrame(onehot_rows)


def mine_association_rules(transactions,
                           min_support=0.0001,
                           min_lift=1.2,
                           min_confidence=0.2,
                           top_n_items=500):
    """
    Run FPGrowth on filtered transactions (max itemset length = 2) and generate association rules.
    Returns a DataFrame of filtered rules.
    """
    # Filter to top items
    filtered_tx, items = filter_top_items(transactions, top_n=top_n_items)
    print(f"Filtered to {len(items)} items, {len(filtered_tx)} transactions")

    # One-hot encode
    onehot = transactions_to_onehot(filtered_tx, items)

    # FPGrowth for itemsets up to length 2
    frequent_itemsets = fpgrowth(
        onehot,
        min_support=min_support,
        use_colnames=True,
        max_len=2
    )
    
    # Association rules
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
    return rules.sort_values('lift', ascending=False)


def main():
    # Load co-purchase edges
    edges_df = load_edges()
    print(f"Loaded {len(edges_df)} edges.")

    # Build transactions
    transactions = build_transactions(edges_df)
    print(f"Built {len(transactions)} transactions.")

    # Mine rules
    rules = mine_association_rules(
        transactions,
        min_support=0.0001,
        min_lift=1.2,
        min_confidence=0.2,
        top_n_items=500
    )

    print("Top 10 association rules (by lift):")
    print(rules.head(10).to_string(index=False))

    # Save to CSV
    out_path = os.path.join(DATA_DIR, 'bundle_rules.csv')
    rules.to_csv(out_path, index=False)
    print(f"Saved {len(rules)} rules to {out_path}")

if __name__ == '__main__':
    main()
