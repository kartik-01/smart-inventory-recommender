import pandas as pd
import re
from datetime import datetime

def process_record(record, products, edges, reviews):
    """
    Given a parsed record, append entries to products, edges, and reviews lists.
    """
    asin = record.get('ASIN')
    # Skip discontinued products
    if record.get('group', '').lower() == 'discontinued product':
        return

    # Products table
    products.append({
        'Id': int(record.get('Id')),
        'ASIN': asin,
        'title': record.get('title', ''),
        'salesrank': record.get('salesrank', ''),
        'group': record.get('group', '')
    })

    # Edges table (co-purchase graph)
    for sim in record.get('similar', []):
        edges.append({'source': asin, 'target': sim})

    # Reviews table
    for line in record.get('review_lines', []):
        # Example: "2000-7-28  customer: A2... rating: 5 votes: 10 helpful: 9"
        parts = line.split()
        date_str = parts[0]
        # Extract rating
        rating_match = re.search(r'rating:\s*(\d+)', line)
        rating = int(rating_match.group(1)) if rating_match else None
        reviews.append({'ASIN': asin, 'date': date_str, 'rating': rating})


def parse_amazon_meta(input_file):
    products = []
    edges = []
    reviews = []
    record = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # Start of a new record
            if line.startswith('Id:'):
                if record:
                    process_record(record, products, edges, reviews)
                    record = {}
                record['Id'] = line.split(':', 1)[1].strip()

            elif line.startswith('ASIN:'):
                record['ASIN'] = line.split(':', 1)[1].strip()

            elif line.strip().startswith('title:'):
                record['title'] = line.split(':', 1)[1].strip()

            elif line.strip().startswith('group:'):
                record['group'] = line.split(':', 1)[1].strip()

            elif line.strip().startswith('salesrank:'):
                record['salesrank'] = line.split(':', 1)[1].strip()

            elif line.strip().startswith('similar:'):
                parts = line.split()
                # e.g. similar: 5  ASIN1 ASIN2 ...
                count = int(parts[1]) if len(parts) > 1 else 0
                record['similar'] = parts[2:2 + count]

            elif line.strip().startswith('categories:'):
                # We skip categories for ingestion
                continue

            elif line.strip().startswith('reviews:'):
                # e.g. reviews: total: 12  downloaded: 12  avg rating: 4.5
                total_match = re.search(r'total:\s*(\d+)', line)
                total = int(total_match.group(1)) if total_match else 0
                record['review_lines'] = []
                record['reviews_to_read'] = total

            elif 'reviews_to_read' in record and record['reviews_to_read'] > 0 and line.strip():
                # Collect review lines
                record['review_lines'].append(line.strip())
                record['reviews_to_read'] -= 1

        # Process the last record if present
        if record:
            process_record(record, products, edges, reviews)

    # Convert to DataFrames
    df_products = pd.DataFrame(products)
    df_edges = pd.DataFrame(edges)
    df_reviews = pd.DataFrame(reviews)

    # Clean and convert types
    df_products['salesrank'] = pd.to_numeric(df_products['salesrank'], errors='coerce')
    median_rank = int(df_products['salesrank'].median())
    df_products['salesrank'] = df_products['salesrank'].fillna(median_rank).astype(int)

    df_reviews['date'] = pd.to_datetime(df_reviews['date'], format='%Y-%m-%d', errors='coerce')

    # Output CSVs
    df_products.to_csv('../data/products.csv', index=False)
    df_edges.to_csv('../data/edges.csv', index=False)
    df_reviews.to_csv('../data/reviews.csv', index=False)

    print(f"Saved: {len(df_products)} products, {len(df_edges)} edges, {len(df_reviews)} reviews to data/ directory")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse Amazon meta data into CSVs')
    parser.add_argument('input_file', help='Path to amazon-meta.txt file')
    args = parser.parse_args()
    parse_amazon_meta(args.input_file)
