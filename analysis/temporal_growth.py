#!/usr/bin/env python3
"""
Moltbook Observatory: Temporal and Growth Analysis
===================================================
Comprehensive analysis of activity patterns, platform growth, and community dynamics.

Sections:
  1. Daily Activity Timeline — Posts/comments per day with cumulative totals
  2. Hourly Activity Patterns — Activity distribution by hour of day
  3. Platform Growth — Total agents and posts from snapshots with growth rates
  4. Response Latency — Time from post to first comment, binned distribution
  5. Community Lifecycle — Submolt activity spans and longevity
  6. Summary Statistics — Overall platform metrics
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import data loaders
from load_data import load_posts, load_comments, load_snapshots


def ensure_output_dir(output_path):
    """Create results directory if needed."""
    results_dir = os.path.dirname(output_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)


def write_section(f, title, content=""):
    """Write a formatted section header and optional content."""
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"{title}\n")
    f.write("=" * 80 + "\n")
    if content:
        f.write(content + "\n")


def analyze_daily_activity(posts_df, comments_df, f):
    """
    SECTION 1: DAILY ACTIVITY TIMELINE

    Parse created_at to datetime, extract date, count posts/comments per day,
    output table with Date, Posts, Comments, Total, Cumulative Posts, Cumulative Comments.
    """
    write_section(f, "SECTION 1: DAILY ACTIVITY TIMELINE")

    # Parse timestamps
    posts_df['created_dt'] = pd.to_datetime(posts_df['created_at'], format='ISO8601', errors='coerce')
    comments_df['created_dt'] = pd.to_datetime(comments_df['created_at'], format='ISO8601', errors='coerce')

    # Extract dates
    posts_df['date'] = posts_df['created_dt'].dt.date
    comments_df['date'] = comments_df['created_dt'].dt.date

    # Count by date
    posts_daily = posts_df.groupby('date').size().rename('Posts')
    comments_daily = comments_df.groupby('date').size().rename('Comments')

    # Combine
    daily_activity = pd.DataFrame({
        'Date': sorted(set(list(posts_daily.index) + list(comments_daily.index)))
    })
    daily_activity['Posts'] = daily_activity['Date'].map(posts_daily).fillna(0).astype(int)
    daily_activity['Comments'] = daily_activity['Date'].map(comments_daily).fillna(0).astype(int)
    daily_activity['Total'] = daily_activity['Posts'] + daily_activity['Comments']
    daily_activity['Cumulative Posts'] = daily_activity['Posts'].cumsum()
    daily_activity['Cumulative Comments'] = daily_activity['Comments'].cumsum()

    # Write table
    f.write("\nDaily Activity Timeline:\n")
    f.write(daily_activity.to_string(index=False))
    f.write("\n")

    return posts_df, comments_df, daily_activity


def analyze_hourly_activity(posts_df, comments_df, f):
    """
    SECTION 2: HOURLY ACTIVITY PATTERNS

    Extract hour from created_at (UTC), count posts/comments per hour,
    output 24-row table with Hour, Posts, Comments, Total.
    Identify peak and trough hours.
    """
    write_section(f, "SECTION 2: HOURLY ACTIVITY PATTERNS")

    # Extract hour
    posts_df['hour'] = posts_df['created_dt'].dt.hour
    comments_df['hour'] = comments_df['created_dt'].dt.hour

    # Count by hour
    posts_hourly = posts_df.groupby('hour').size().rename('Posts')
    comments_hourly = comments_df.groupby('hour').size().rename('Comments')

    # Create complete 24-hour table
    hourly_activity = pd.DataFrame({
        'Hour': range(24)
    })
    hourly_activity['Posts'] = hourly_activity['Hour'].map(posts_hourly).fillna(0).astype(int)
    hourly_activity['Comments'] = hourly_activity['Hour'].map(comments_hourly).fillna(0).astype(int)
    hourly_activity['Total'] = hourly_activity['Posts'] + hourly_activity['Comments']

    # Write table
    f.write("\nHourly Activity Distribution (UTC):\n")
    f.write(hourly_activity.to_string(index=False))
    f.write("\n")

    # Identify peak and trough
    peak_hour = hourly_activity.loc[hourly_activity['Total'].idxmax()]
    trough_hour = hourly_activity.loc[hourly_activity['Total'].idxmin()]

    f.write(f"\nPeak Hour: {int(peak_hour['Hour']):02d}:00 with {int(peak_hour['Total'])} activities\n")
    f.write(f"Trough Hour: {int(trough_hour['Hour']):02d}:00 with {int(trough_hour['Total'])} activities\n")

    return hourly_activity


def analyze_platform_growth(f):
    """
    SECTION 3: PLATFORM GROWTH

    Load snapshots: show total_agents, total_posts per snapshot timestamp.
    Calculate growth rate between consecutive snapshots.
    Identify inflection point (highest absolute growth).
    """
    write_section(f, "SECTION 3: PLATFORM GROWTH")

    try:
        snapshots_df = load_snapshots()

        if snapshots_df.empty:
            f.write("\nNo snapshot data available.\n")
            return None

        # Parse timestamp
        snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'], format='ISO8601', errors='coerce')
        snapshots_df = snapshots_df.sort_values('timestamp').reset_index(drop=True)

        # Select relevant columns
        growth_df = snapshots_df[['timestamp', 'total_agents', 'total_posts']].copy()

        # Calculate growth rates
        growth_df['agent_growth'] = growth_df['total_agents'].diff().fillna(0).astype(int)
        growth_df['post_growth'] = growth_df['total_posts'].diff().fillna(0).astype(int)

        # Write table
        f.write("\nPlatform Growth Over Time:\n")
        f.write(growth_df.to_string(index=False))
        f.write("\n")

        # Identify inflection points
        if len(growth_df) > 1:
            max_agent_growth_idx = growth_df['agent_growth'].idxmax()
            max_post_growth_idx = growth_df['post_growth'].idxmax()

            f.write(f"\nInflection Points:\n")
            f.write(f"  Highest agent growth: {int(growth_df.loc[max_agent_growth_idx, 'agent_growth'])} agents on "
                   f"{growth_df.loc[max_agent_growth_idx, 'timestamp']}\n")
            f.write(f"  Highest post growth: {int(growth_df.loc[max_post_growth_idx, 'post_growth'])} posts on "
                   f"{growth_df.loc[max_post_growth_idx, 'timestamp']}\n")

        return growth_df
    except Exception as e:
        f.write(f"\nSnapshot data not available: {e}\n")
        return None


def analyze_response_latency(posts_df, comments_df, f):
    """
    SECTION 4: RESPONSE LATENCY

    For each post, find earliest comment (min comment created_at for that post_id).
    Compute latency = first_comment_time - post_created_at.
    Report: median, mean, p10, p25, p75, p90 latency.
    Bin into: 0-10s, 10-30s, 30-60s, 1-5min, 5-30min, 30-60min, 1-6h, 6-24h, 24h+
    Output distribution table.
    """
    write_section(f, "SECTION 4: RESPONSE LATENCY")

    # Find first comment for each post
    first_comments = comments_df.groupby('post_id')['created_dt'].min().reset_index()
    first_comments.columns = ['post_id', 'first_comment_dt']

    # Merge with posts
    posts_with_latency = posts_df[['id', 'created_dt']].copy()
    posts_with_latency.columns = ['post_id', 'post_created_dt']
    posts_with_latency = posts_with_latency.merge(first_comments, on='post_id', how='left')

    # Calculate latency (only for posts with comments)
    posts_with_latency = posts_with_latency.dropna(subset=['first_comment_dt'])
    posts_with_latency['latency_seconds'] = (
        posts_with_latency['first_comment_dt'] - posts_with_latency['post_created_dt']
    ).dt.total_seconds()

    # Remove negative latencies (data issues)
    posts_with_latency = posts_with_latency[posts_with_latency['latency_seconds'] >= 0]

    if len(posts_with_latency) == 0:
        f.write("\nNo posts with comments found for latency analysis.\n")
        return None

    latencies = posts_with_latency['latency_seconds']

    # Statistics
    f.write(f"\nResponse Latency Statistics (seconds):\n")
    f.write(f"  Count: {len(latencies)}\n")
    f.write(f"  Mean: {latencies.mean():.1f}s\n")
    f.write(f"  Median: {latencies.median():.1f}s\n")
    f.write(f"  Std Dev: {latencies.std():.1f}s\n")
    f.write(f"  Min: {latencies.min():.1f}s\n")
    f.write(f"  Max: {latencies.max():.1f}s\n")
    f.write(f"\nPercentiles:\n")
    f.write(f"  p10: {latencies.quantile(0.10):.1f}s\n")
    f.write(f"  p25: {latencies.quantile(0.25):.1f}s\n")
    f.write(f"  p50: {latencies.quantile(0.50):.1f}s\n")
    f.write(f"  p75: {latencies.quantile(0.75):.1f}s\n")
    f.write(f"  p90: {latencies.quantile(0.90):.1f}s\n")

    # Bin latencies into categories
    bins = [
        (0, 10, '0-10s'),
        (10, 30, '10-30s'),
        (30, 60, '30-60s'),
        (60, 300, '1-5min'),
        (300, 1800, '5-30min'),
        (1800, 3600, '30-60min'),
        (3600, 21600, '1-6h'),
        (21600, 86400, '6-24h'),
        (86400, float('inf'), '24h+')
    ]

    latency_dist = []
    for lower, upper, label in bins:
        count = ((latencies >= lower) & (latencies < upper)).sum()
        percentage = 100 * count / len(latencies)
        latency_dist.append({
            'Bin': label,
            'Count': count,
            'Percentage': f"{percentage:.2f}%"
        })

    latency_dist_df = pd.DataFrame(latency_dist)
    f.write(f"\nLatency Distribution:\n")
    f.write(latency_dist_df.to_string(index=False))
    f.write("\n")

    return latency_dist_df


def analyze_community_lifecycle(posts_df, f):
    """
    SECTION 5: COMMUNITY LIFECYCLE

    For each submolt, find: first_post_time, last_post_time, total_posts, lifespan_hours.
    Report: total submolts, median lifespan, % dead within 1 hour (only 1 post ever).
    Show top 10 longest-lived submolts and top 10 largest by post count.
    """
    write_section(f, "SECTION 5: COMMUNITY LIFECYCLE")

    if 'submolt' not in posts_df.columns:
        f.write("\nSubmolt column not found in posts data.\n")
        return None

    # Analyze submolt lifecycle
    submolt_stats = []
    for submolt, group in posts_df.groupby('submolt'):
        first_post = group['created_dt'].min()
        last_post = group['created_dt'].max()
        total_posts = len(group)

        if pd.notna(first_post) and pd.notna(last_post):
            lifespan_hours = (last_post - first_post).total_seconds() / 3600
        else:
            lifespan_hours = 0

        submolt_stats.append({
            'Submolt': submolt,
            'First Post': first_post,
            'Last Post': last_post,
            'Total Posts': total_posts,
            'Lifespan (hours)': lifespan_hours
        })

    lifecycle_df = pd.DataFrame(submolt_stats)

    # Summary statistics
    total_submolts = len(lifecycle_df)
    median_lifespan = lifecycle_df['Lifespan (hours)'].median()
    dead_within_1h = (lifecycle_df['Total Posts'] == 1).sum()
    pct_dead = 100 * dead_within_1h / total_submolts if total_submolts > 0 else 0

    f.write(f"\nCommunity Lifecycle Summary:\n")
    f.write(f"  Total submolts: {total_submolts}\n")
    f.write(f"  Median lifespan: {median_lifespan:.2f} hours\n")
    f.write(f"  Communities with only 1 post: {dead_within_1h} ({pct_dead:.2f}%)\n")

    # Top 10 longest-lived
    top_long = lifecycle_df.nlargest(10, 'Lifespan (hours)')[['Submolt', 'Lifespan (hours)', 'Total Posts']]
    f.write(f"\nTop 10 Longest-Lived Submolts:\n")
    f.write(top_long.to_string(index=False))
    f.write("\n")

    # Top 10 largest by post count
    top_large = lifecycle_df.nlargest(10, 'Total Posts')[['Submolt', 'Total Posts', 'Lifespan (hours)']]
    f.write(f"\nTop 10 Largest Submolts (by Post Count):\n")
    f.write(top_large.to_string(index=False))
    f.write("\n")

    return lifecycle_df


def analyze_summary_statistics(posts_df, comments_df, f):
    """
    SECTION 6: SUMMARY STATISTICS

    Total posts, comments, agents, submolts.
    Date range.
    Posts per agent (mean, median).
    Comments per post (mean, median).
    """
    write_section(f, "SECTION 6: SUMMARY STATISTICS")

    total_posts = len(posts_df)
    total_comments = len(comments_df)

    unique_agents = posts_df['agent_id'].nunique()
    unique_submolts = posts_df['submolt'].nunique() if 'submolt' in posts_df.columns else 0

    # Date range
    if len(posts_df) > 0:
        post_date_range = (posts_df['created_dt'].min(), posts_df['created_dt'].max())
    else:
        post_date_range = (None, None)

    if len(comments_df) > 0:
        comment_date_range = (comments_df['created_dt'].min(), comments_df['created_dt'].max())
    else:
        comment_date_range = (None, None)

    # Posts per agent
    posts_per_agent = posts_df.groupby('agent_id').size()
    posts_per_agent_mean = posts_per_agent.mean()
    posts_per_agent_median = posts_per_agent.median()

    # Comments per post
    comments_per_post = comments_df.groupby('post_id').size()
    comments_per_post_mean = comments_per_post.mean()
    comments_per_post_median = comments_per_post.median()

    f.write(f"\nPlatform Overview:\n")
    f.write(f"  Total posts: {total_posts:,}\n")
    f.write(f"  Total comments: {total_comments:,}\n")
    f.write(f"  Unique agents: {unique_agents:,}\n")
    f.write(f"  Unique submolts: {unique_submolts:,}\n")

    f.write(f"\nDate Ranges:\n")
    if post_date_range[0]:
        f.write(f"  Posts: {post_date_range[0]} to {post_date_range[1]}\n")
    if comment_date_range[0]:
        f.write(f"  Comments: {comment_date_range[0]} to {comment_date_range[1]}\n")

    f.write(f"\nAgent Activity:\n")
    f.write(f"  Posts per agent (mean): {posts_per_agent_mean:.2f}\n")
    f.write(f"  Posts per agent (median): {posts_per_agent_median:.0f}\n")
    f.write(f"  Posts per agent (min): {posts_per_agent.min()}\n")
    f.write(f"  Posts per agent (max): {posts_per_agent.max()}\n")

    f.write(f"\nComment Activity:\n")
    f.write(f"  Comments per post (mean): {comments_per_post_mean:.2f}\n")
    f.write(f"  Comments per post (median): {comments_per_post_median:.0f}\n")
    f.write(f"  Comments per post (min): {comments_per_post.min()}\n")
    f.write(f"  Comments per post (max): {comments_per_post.max()}\n")


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Temporal and Growth Analysis for Moltbook Observatory"
    )
    parser.add_argument(
        '--output',
        default='results/temporal_growth_report.txt',
        help='Output file path (default: results/temporal_growth_report.txt)'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    ensure_output_dir(args.output)

    print(f"Loading data...")
    posts_df = load_posts()
    comments_df = load_comments()

    print(f"Running analyses and writing to {args.output}...")

    with open(args.output, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("MOLTBOOK OBSERVATORY: TEMPORAL AND GROWTH ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("\n")

        # Run all analyses
        posts_df, comments_df, daily_activity = analyze_daily_activity(posts_df, comments_df, f)
        hourly_activity = analyze_hourly_activity(posts_df, comments_df, f)
        growth_df = analyze_platform_growth(f)
        latency_dist = analyze_response_latency(posts_df, comments_df, f)
        lifecycle_df = analyze_community_lifecycle(posts_df, f)
        analyze_summary_statistics(posts_df, comments_df, f)

        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("=" * 80 + "\n")

    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
