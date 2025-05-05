#!/usr/bin/env python3
"""
Compute reviewer-level profiling features by extracting review info directly from Amazon metadata or a detailed reviews CSV.

Usage:
  # If you have a detailed reviews CSV:
  python src/reviewer_profiling.py --reviews data/reviews.csv --output data/reviewer_features.csv

  # Otherwise, parse the raw Amazon metadata file:
  python src/reviewer_profiling.py --meta data/amazon-meta.txt[.gz] --output data/reviewer_features.csv
"""
import argparse
import gzip
import re
import pandas as pd
import numpy as np


def parse_reviews_from_meta(meta_path):
    """
    Parse the raw Amazon metadata file to extract review records.
    Returns a DataFrame with columns: ASIN, date, customer, rating, votes, helpful
    """
    reviews = []
    open_fn = gzip.open if meta_path.endswith('.gz') else open
    with open_fn(meta_path, 'rt', encoding='utf-8', errors='ignore') as f:
        asin = None
        for line in f:
            line = line.strip()
            if line.startswith('ASIN:'):
                asin = line.split(':', 1)[1].strip()
            elif not asin:
                continue
            else:
                m = re.match(
                    r"^(\d{4}-\d{1,2}-\d{1,2})\s+cutomer:\s*([A-Z0-9]+)\s+rating:\s*(\d+)\s+votes:\s*(\d+)\s+helpful:\s*(\d+)",
                    line, re.IGNORECASE
                )
                if m:
                    date, customer, rating, votes, helpful = m.groups()
                    reviews.append({
                        'ASIN': asin,
                        'date': date,
                        'customer': customer,
                        'rating': int(rating),
                        'votes': int(votes),
                        'helpful': int(helpful)
                    })
    df = pd.DataFrame(reviews)
    if df.empty:
        raise ValueError(f"No reviews parsed from {meta_path}")
    print(f"Parsed {len(df)} reviews from metadata.")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of raw reviews (with columns customer, date, rating, votes, helpful),
    compute per-customer profiling features.
    """
    # Standardize column names
    df = df.rename(columns={'customer': 'customer_id', 'date': 'review_date'})
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')

    # Group by reviewer
    g = df.groupby('customer_id')

    # Core stats
    features = pd.DataFrame({
        'total_reviews': g.size(),
        'avg_rating': g['rating'].mean(),
        'std_rating': g['rating'].std().fillna(0)
    })

    # Helpfulness ratio
    total_helpful = g['helpful'].sum()
    total_votes = g['votes'].sum()
    features['helpfulness_ratio'] = total_helpful / total_votes.replace(0, np.nan)

    # Active span
    first = g['review_date'].min()
    last = g['review_date'].max()
    features['active_days_span'] = (last - first).dt.days

    # Interval stats function
    def interval_stats(dates: pd.Series) -> pd.Series:
        dates = dates.sort_values()
        diffs = dates.diff().dt.days.dropna()
        if diffs.empty:
            return pd.Series({'median_interval': np.nan, 'burstiness': np.nan})
        median_int = diffs.median()
        mean_int = diffs.mean()
        std_int = diffs.std()
        burst = std_int / mean_int if mean_int else np.nan
        return pd.Series({'median_interval': median_int, 'burstiness': burst})

    # Apply and unstack to get separate columns
    ivals = g['review_date'].apply(interval_stats).unstack()
    features = features.join(ivals)

    # Reviews per month
    features['reviews_per_month'] = (
        features['total_reviews'] /
        (features['active_days_span'] / 30).replace(0, np.nan)
    )

    return features.reset_index()


def main():
    parser = argparse.ArgumentParser(description='Reviewer profiling from reviews CSV or Amazon metadata')
    parser.add_argument('--reviews', type=str, default=None,
                        help='Path to a detailed reviews CSV')
    parser.add_argument('--meta', type=str, default=None,
                        help='Path to raw amazon-meta.txt or .gz')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to write reviewer features CSV')
    args = parser.parse_args()

    if args.reviews and args.meta:
        parser.error('Specify either --reviews or --meta, not both')
    if args.reviews:
        df = pd.read_csv(args.reviews)
    elif args.meta:
        df = parse_reviews_from_meta(args.meta)
    else:
        parser.error('Must provide either --reviews or --meta input')

    feats = compute_features(df)
    feats.to_csv(args.output, index=False)
    print(f"Wrote {len(feats)} reviewer feature records to {args.output}")


if __name__ == '__main__':
    main()
