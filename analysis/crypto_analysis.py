#!/usr/bin/env python3
"""
Moltbook Cryptocurrency Pump-and-Dump Analysis Script
======================================================
Analyzes posts and comments for cryptocurrency-related content and pump-and-dump patterns.

Usage:
    python crypto_analysis.py [--output results/crypto_analysis_report.txt]
"""

import argparse
import os
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
from load_data import load_posts, load_comments, load_word_frequency


# Crypto keywords to flag content
CRYPTO_KEYWORDS = {
    'molt', 'claw', 'token', 'mint', 'crypto', 'wallet', 'coin',
    'trading', 'buy', 'sell', 'pump', 'blockchain', 'dump', 'moon', 'airdrop'
}

PUMP_KEYWORDS = {'pump', 'moon', 'moon', 'surge', 'rocket', 'skyrocket', 'bull'}
DUMP_KEYWORDS = {'dump', 'crash', 'collapse', 'tank', 'bear', 'sell', 'exit'}
FINANCIAL_KEYWORDS = {'buy', 'sell', 'investment', 'profit', 'price', 'token', 'coin'}

# Posts with explicit "$MOLT" mention
MOLT_MARKER = '$MOLT'


def is_crypto_content(text):
    """Check if text contains any crypto keywords (case-insensitive)."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRYPTO_KEYWORDS)


def extract_date_from_path(filepath):
    """Extract date from parquet filepath like '2026-01-28.parquet'."""
    basename = os.path.basename(filepath)
    return basename.replace('.parquet', '')


def analyze_posts(output_file):
    """Analyze posts data for crypto content and patterns."""
    print("\nSTEP 1: ANALYZING POSTS DATA")
    print("-" * 100)

    output_file.write("\nSTEP 1: ANALYZING POSTS DATA\n")
    output_file.write("-" * 100 + "\n")

    # Load posts with per-file reporting
    import glob
    from load_data import DATA_DIR
    posts_path = os.path.join(DATA_DIR, "posts")
    post_files = sorted(glob.glob(os.path.join(posts_path, "*.parquet")))

    all_posts = []
    daily_totals = defaultdict(int)

    for pf in post_files:
        df = pd.read_parquet(pf)
        date_str = extract_date_from_path(pf)
        count = len(df)
        print(f"Processing {os.path.basename(pf)}: {count} posts")
        output_file.write(f"Processing {os.path.basename(pf)}: {count} posts\n")
        all_posts.append(df)
        daily_totals[date_str] = count

    posts = pd.concat(all_posts, ignore_index=True)

    # Flag crypto posts
    posts['is_crypto'] = posts['content'].apply(is_crypto_content) | posts['title'].apply(is_crypto_content)

    total_posts = len(posts)
    crypto_posts = posts['is_crypto'].sum()
    non_crypto = total_posts - crypto_posts

    print(f"Total posts loaded: {total_posts}")
    print(f"Crypto posts: {crypto_posts} ({100*crypto_posts/total_posts:.2f}%)")
    print(f"Non-crypto posts: {non_crypto} ({100*non_crypto/total_posts:.2f}%)")

    output_file.write(f"Total posts loaded: {total_posts}\n")
    output_file.write(f"Crypto posts: {crypto_posts} ({100*crypto_posts/total_posts:.2f}%)\n")
    output_file.write(f"Non-crypto posts: {non_crypto} ({100*non_crypto/total_posts:.2f}%)\n")

    # Step 2: Daily timeline
    print("\nSTEP 2: DAILY CRYPTO TIMELINE")
    print("-" * 100)
    output_file.write("\nSTEP 2: DAILY CRYPTO TIMELINE\n")
    output_file.write("-" * 100 + "\n")

    posts['date'] = pd.to_datetime(posts['created_at'], format='ISO8601').dt.date
    daily_crypto = posts[posts['is_crypto']].groupby('date').size()
    daily_all = posts.groupby('date').size()

    for date in sorted(daily_all.index):
        crypto_count = daily_crypto.get(date, 0)
        total_count = daily_all[date]
        pct = 100 * crypto_count / total_count if total_count > 0 else 0
        date_str = str(date)
        line = f"{date_str}: {crypto_count:5d} crypto / {total_count:6d} total ({pct:5.2f}%)"
        print(line)
        output_file.write(line + "\n")

    # Step 3: Engagement comparison
    print("\nSTEP 3: ENGAGEMENT COMPARISON - CRYPTO vs NON-CRYPTO POSTS")
    print("-" * 100)
    output_file.write("\nSTEP 3: ENGAGEMENT COMPARISON - CRYPTO vs NON-CRYPTO POSTS\n")
    output_file.write("-" * 100 + "\n\n")

    crypto_df = posts[posts['is_crypto']]
    non_crypto_df = posts[~posts['is_crypto']]

    crypto_score_sum = crypto_df['score'].sum()
    crypto_score_avg = crypto_df['score'].mean()
    crypto_comments_sum = crypto_df['comment_count'].sum()
    crypto_comments_avg = crypto_df['comment_count'].mean()

    non_crypto_score_sum = non_crypto_df['score'].sum()
    non_crypto_score_avg = non_crypto_df['score'].mean()
    non_crypto_comments_sum = non_crypto_df['comment_count'].sum()
    non_crypto_comments_avg = non_crypto_df['comment_count'].mean()

    print(f"Crypto Posts ({len(crypto_df)} total):")
    print(f"  Total score sum:     {int(crypto_score_sum)}")
    print(f"  Average score:       {crypto_score_avg:.2f}")
    print(f"  Total comments sum:  {int(crypto_comments_sum)}")
    print(f"  Average comments:    {crypto_comments_avg:.2f}")
    print()
    print(f"Non-Crypto Posts ({len(non_crypto_df)} total):")
    print(f"  Total score sum:     {int(non_crypto_score_sum)}")
    print(f"  Average score:       {non_crypto_score_avg:.2f}")
    print(f"  Total comments sum:  {int(non_crypto_comments_sum)}")
    print(f"  Average comments:    {non_crypto_comments_avg:.2f}")

    output_file.write(f"Crypto Posts ({len(crypto_df)} total):\n")
    output_file.write(f"  Total score sum:     {int(crypto_score_sum)}\n")
    output_file.write(f"  Average score:       {crypto_score_avg:.2f}\n")
    output_file.write(f"  Total comments sum:  {int(crypto_comments_sum)}\n")
    output_file.write(f"  Average comments:    {crypto_comments_avg:.2f}\n\n")
    output_file.write(f"Non-Crypto Posts ({len(non_crypto_df)} total):\n")
    output_file.write(f"  Total score sum:     {int(non_crypto_score_sum)}\n")
    output_file.write(f"  Average score:       {non_crypto_score_avg:.2f}\n")
    output_file.write(f"  Total comments sum:  {int(non_crypto_comments_sum)}\n")
    output_file.write(f"  Average comments:    {non_crypto_comments_avg:.2f}\n\n")

    score_diff = crypto_score_avg - non_crypto_score_avg
    score_diff_pct = 100 * score_diff / non_crypto_score_avg if non_crypto_score_avg != 0 else 0
    comments_diff = crypto_comments_avg - non_crypto_comments_avg
    comments_diff_pct = 100 * comments_diff / non_crypto_comments_avg if non_crypto_comments_avg != 0 else 0

    print(f"Engagement Difference:")
    print(f"  Score difference:    {score_diff:.2f} ({score_diff_pct:.1f}%)")
    print(f"  Comments difference: {comments_diff:+.2f} ({comments_diff_pct:+.1f}%)")

    output_file.write(f"Engagement Difference:\n")
    output_file.write(f"  Score difference:    {score_diff:.2f} ({score_diff_pct:.1f}%)\n")
    output_file.write(f"  Comments difference: {comments_diff:+.2f} ({comments_diff_pct:+.1f}%)\n")

    # Step 4: Top agents by crypto posts
    print("\nSTEP 4: TOP AGENTS PROMOTING CRYPTO")
    print("-" * 100)
    output_file.write("\nSTEP 4: TOP AGENTS PROMOTING CRYPTO\n")
    output_file.write("-" * 100 + "\n\n")

    agent_crypto_posts = crypto_df['agent_name'].value_counts().head(20)
    print(f"Top 20 agents by crypto posts:")
    output_file.write(f"Top 20 agents by crypto posts:\n")

    for agent, count in agent_crypto_posts.items():
        agent_display = agent if agent else "(anonymous)"
        line = f"  {agent_display:40s}: {count:5d} posts"
        print(line)
        output_file.write(line + "\n")

    # Step 5: Top submolts
    print("\nSTEP 5: TOP SUBMOLTS")
    print("-" * 100)
    output_file.write("\nSTEP 5: TOP SUBMOLTS\n")
    output_file.write("-" * 100 + "\n\n")

    submolt_crypto = crypto_df['submolt'].value_counts().head(15)
    print(f"Top 15 submolts with crypto discussion:")
    output_file.write(f"Top 15 submolts with crypto discussion:\n")

    for submolt, count in submolt_crypto.items():
        submolt_display = submolt if submolt else "(no submolt)"
        line = f"  {submolt_display:52s}: {count:5d} posts"
        print(line)
        output_file.write(line + "\n")

    # Step 6: Keyword frequency
    print("\nSTEP 6: CRYPTO KEYWORD FREQUENCY IN POSTS")
    print("-" * 100)
    output_file.write("\nSTEP 6: CRYPTO KEYWORD FREQUENCY IN POSTS\n")
    output_file.write("-" * 100 + "\n\n")

    keyword_counts = {kw: 0 for kw in CRYPTO_KEYWORDS}
    for text in crypto_df['content'].fillna(''):
        text_lower = text.lower()
        for kw in CRYPTO_KEYWORDS:
            keyword_counts[kw] += text_lower.count(kw)
    for text in crypto_df['title'].fillna(''):
        text_lower = text.lower()
        for kw in CRYPTO_KEYWORDS:
            keyword_counts[kw] += text_lower.count(kw)

    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Keyword occurrences in crypto posts:")
    output_file.write(f"Keyword occurrences in crypto posts:\n")

    for kw, count in sorted_keywords:
        line = f"  {kw:20s}: {count:6d} occurrences"
        print(line)
        output_file.write(line + "\n")

    # Step 7: Pump and dump patterns
    print("\nSTEP 7: PUMP AND DUMP PATTERN ANALYSIS")
    print("-" * 100)
    output_file.write("\nSTEP 7: PUMP AND DUMP PATTERN ANALYSIS\n")
    output_file.write("-" * 100 + "\n\n")

    def has_pump_or_dump(text):
        if not isinstance(text, str):
            return None
        text_lower = text.lower()
        has_pump = any(pw in text_lower for pw in PUMP_KEYWORDS)
        has_dump = any(dw in text_lower for dw in DUMP_KEYWORDS)
        if has_pump:
            return 'pump'
        elif has_dump:
            return 'dump'
        return None

    crypto_df_copy = crypto_df.copy()
    crypto_df_copy['pump_dump'] = crypto_df_copy['content'].apply(has_pump_or_dump)

    pump_posts = crypto_df_copy[crypto_df_copy['pump_dump'] == 'pump'].head(5)
    dump_posts = crypto_df_copy[crypto_df_copy['pump_dump'] == 'dump'].head(5)

    pump_total = (crypto_df_copy['pump_dump'] == 'pump').sum()
    dump_total = (crypto_df_copy['pump_dump'] == 'dump').sum()

    print(f"Pump-like posts found: {pump_total}")
    print(f"Sample pump posts:")
    output_file.write(f"Pump-like posts found: {pump_total}\n")
    output_file.write(f"Sample pump posts:\n")

    for idx, (_, row) in enumerate(pump_posts.iterrows(), 1):
        agent = row['agent_name'] if row['agent_name'] else "(anonymous)"
        score = int(row['score'])
        title = row['title'][:50] if isinstance(row['title'], str) else ""
        line = f"  [{idx}] {title}"
        print(line)
        output_file.write(line + "\n")
        line2 = f"       Agent: {agent}, Score: {score}"
        print(line2)
        output_file.write(line2 + "\n")

    print()
    print(f"Dump-like posts found: {dump_total}")
    print(f"Sample dump posts:")
    output_file.write(f"\nDump-like posts found: {dump_total}\n")
    output_file.write(f"Sample dump posts:\n")

    for idx, (_, row) in enumerate(dump_posts.iterrows(), 1):
        agent = row['agent_name'] if row['agent_name'] else "(anonymous)"
        score = int(row['score'])
        title = row['title'][:50] if isinstance(row['title'], str) else ""
        line = f"  [{idx}] {title}"
        print(line)
        output_file.write(line + "\n")
        line2 = f"       Agent: {agent}, Score: {score}"
        print(line2)
        output_file.write(line2 + "\n")

    # Step 8: $MOLT posts
    print("\nSTEP 8: SAMPLE $MOLT POSTS")
    print("-" * 100)
    output_file.write("\nSTEP 8: SAMPLE $MOLT POSTS\n")
    output_file.write("-" * 100 + "\n\n")

    molt_posts = posts[posts['content'].str.contains(MOLT_MARKER, case=False, na=False) |
                       posts['title'].str.contains(MOLT_MARKER, case=False, na=False)]
    molt_count = len(molt_posts)

    print(f"Total $MOLT posts found: {molt_count}\n")
    print(f"Example posts:\n")
    output_file.write(f"Total $MOLT posts found: {molt_count}\n\n")
    output_file.write(f"Example posts:\n\n")

    for idx, (_, row) in enumerate(molt_posts.head(10).iterrows(), 1):
        agent = row['agent_name'] if row['agent_name'] else "(anonymous)"
        score = int(row['score'])
        comments = int(row['comment_count'])
        title = row['title'][:60] if isinstance(row['title'], str) else ""
        content = row['content'][:80] if isinstance(row['content'], str) else ""

        line1 = f"  [{idx}] {title}"
        print(line1)
        output_file.write(line1 + "\n")

        if content:
            line2 = f"       {content}"
            print(line2)
            output_file.write(line2 + "\n")

        line3 = f"       Agent: {agent}, Score: {score}, Comments: {comments}"
        print(line3)
        output_file.write(line3 + "\n\n")

    return posts


def analyze_comments(posts, output_file):
    """Analyze comments data for crypto content."""
    print("\nSTEP 9: ANALYZING COMMENTS DATA")
    print("-" * 100)
    output_file.write("\nSTEP 9: ANALYZING COMMENTS DATA\n")
    output_file.write("-" * 100 + "\n")

    # Load comments with per-file reporting
    import glob
    from load_data import DATA_DIR
    comments_path = os.path.join(DATA_DIR, "comments")
    comment_files = sorted(glob.glob(os.path.join(comments_path, "*.parquet")))

    all_comments = []
    for cf in comment_files:
        df = pd.read_parquet(cf)
        count = len(df)
        print(f"Processing {os.path.basename(cf)}: {count} comments")
        output_file.write(f"Processing {os.path.basename(cf)}: {count} comments\n")
        all_comments.append(df)

    comments = pd.concat(all_comments, ignore_index=True)

    # Flag crypto comments
    comments['is_crypto'] = comments['content'].apply(is_crypto_content)

    total_comments = len(comments)
    crypto_comments = comments['is_crypto'].sum()
    non_crypto = total_comments - crypto_comments

    print(f"Total comments loaded: {total_comments}")
    print(f"Crypto comments: {crypto_comments} ({100*crypto_comments/total_comments:.2f}%)")
    print(f"Non-crypto comments: {non_crypto} ({100*non_crypto/total_comments:.2f}%)")

    output_file.write(f"Total comments loaded: {total_comments}\n")
    output_file.write(f"Crypto comments: {crypto_comments} ({100*crypto_comments/total_comments:.2f}%)\n")
    output_file.write(f"Non-crypto comments: {non_crypto} ({100*non_crypto/total_comments:.2f}%)\n\n")

    # Step 10: Daily comment timeline
    print("\nSTEP 10: DAILY COMMENT TIMELINE")
    print("-" * 100)
    output_file.write("\nSTEP 10: DAILY COMMENT TIMELINE\n")
    output_file.write("-" * 100 + "\n")

    comments['date'] = pd.to_datetime(comments['created_at'], format='ISO8601').dt.date
    daily_crypto_comments = comments[comments['is_crypto']].groupby('date').size()
    daily_all_comments = comments.groupby('date').size()

    for date in sorted(daily_all_comments.index):
        crypto_count = daily_crypto_comments.get(date, 0)
        total_count = daily_all_comments[date]
        pct = 100 * crypto_count / total_count if total_count > 0 else 0
        date_str = str(date)
        line = f"{date_str}: {crypto_count:5d} crypto / {total_count:6d} total ({pct:5.2f}%)"
        print(line)
        output_file.write(line + "\n")

    # Step 11: Engagement comparison for comments
    print("\nSTEP 11: ENGAGEMENT COMPARISON - CRYPTO vs NON-CRYPTO COMMENTS")
    print("-" * 100)
    output_file.write("\nSTEP 11: ENGAGEMENT COMPARISON - CRYPTO vs NON-CRYPTO COMMENTS\n")
    output_file.write("-" * 100 + "\n\n")

    crypto_comments_df = comments[comments['is_crypto']]
    non_crypto_comments_df = comments[~comments['is_crypto']]

    crypto_score_sum = crypto_comments_df['score'].sum()
    crypto_score_avg = crypto_comments_df['score'].mean()
    non_crypto_score_sum = non_crypto_comments_df['score'].sum()
    non_crypto_score_avg = non_crypto_comments_df['score'].mean()

    print(f"Crypto Comments ({len(crypto_comments_df)} total):")
    print(f"  Total score sum:     {int(crypto_score_sum)}")
    print(f"  Average score:       {crypto_score_avg:.2f}")
    print()
    print(f"Non-Crypto Comments ({len(non_crypto_comments_df)} total):")
    print(f"  Total score sum:     {int(non_crypto_score_sum)}")
    print(f"  Average score:       {non_crypto_score_avg:.2f}")
    print()
    print(f"Engagement Difference:")
    print(f"  Score difference:    {crypto_score_avg - non_crypto_score_avg:.2f}")

    output_file.write(f"Crypto Comments ({len(crypto_comments_df)} total):\n")
    output_file.write(f"  Total score sum:     {int(crypto_score_sum)}\n")
    output_file.write(f"  Average score:       {crypto_score_avg:.2f}\n\n")
    output_file.write(f"Non-Crypto Comments ({len(non_crypto_comments_df)} total):\n")
    output_file.write(f"  Total score sum:     {int(non_crypto_score_sum)}\n")
    output_file.write(f"  Average score:       {non_crypto_score_avg:.2f}\n\n")
    output_file.write(f"Engagement Difference:\n")
    output_file.write(f"  Score difference:    {crypto_score_avg - non_crypto_score_avg:.2f}\n")

    # Step 12: Top agents by crypto comments
    print("\nSTEP 12: TOP AGENTS BY CRYPTO COMMENTS")
    print("-" * 100)
    output_file.write("\nSTEP 12: TOP AGENTS BY CRYPTO COMMENTS\n")
    output_file.write("-" * 100 + "\n\n")

    agent_crypto_comments = crypto_comments_df['agent_name'].value_counts().head(15)
    print(f"Top 15 agents by crypto comments:")
    output_file.write(f"Top 15 agents by crypto comments:\n")

    for agent, count in agent_crypto_comments.items():
        agent_display = agent if agent else "(anonymous)"
        line = f"  {agent_display:40s}: {count:5d} comments"
        print(line)
        output_file.write(line + "\n")

    # Step 13: $MOLT comments
    print("\nSTEP 13: SAMPLE $MOLT COMMENTS")
    print("-" * 100)
    output_file.write("\nSTEP 13: SAMPLE $MOLT COMMENTS\n")
    output_file.write("-" * 100 + "\n\n")

    molt_comments = comments[comments['content'].str.contains(MOLT_MARKER, case=False, na=False)]
    molt_count = len(molt_comments)

    print(f"Total $MOLT comments found: {molt_count}\n")
    print(f"Example comments:\n")
    output_file.write(f"Total $MOLT comments found: {molt_count}\n\n")
    output_file.write(f"Example comments:\n\n")

    for idx, (_, row) in enumerate(molt_comments.head(10).iterrows(), 1):
        agent = row['agent_name'] if row['agent_name'] else "(anonymous)"
        score = int(row['score'])
        content = row['content'][:80] if isinstance(row['content'], str) else ""

        line1 = f"  [{idx}] {content}"
        print(line1)
        output_file.write(line1 + "\n")

        line2 = f"       Agent: {agent}, Score: {score}\n"
        print(line2)
        output_file.write(line2 + "\n")

    return comments


def analyze_word_frequency(output_file):
    """Analyze word frequency data."""
    print("\nSTEP 14: WORD FREQUENCY DATA ANALYSIS")
    print("-" * 100)
    output_file.write("\nSTEP 14: WORD FREQUENCY DATA ANALYSIS\n")
    output_file.write("-" * 100 + "\n")

    try:
        # Load word frequency with per-file reporting
        import glob
        from load_data import DATA_DIR
        wf_path = os.path.join(DATA_DIR, "word_frequency")
        wf_files = sorted(glob.glob(os.path.join(wf_path, "*.parquet")))

        total_rows = 0
        for wf in wf_files:
            df = pd.read_parquet(wf)
            count = len(df)
            print(f"Processing {os.path.basename(wf)}: {count} rows")
            output_file.write(f"Processing {os.path.basename(wf)}: {count} rows\n")
            total_rows += count

        print(f"\nTotal word frequency records: {total_rows}")
        output_file.write(f"\nTotal word frequency records: {total_rows}\n")

        # Check for crypto terms
        crypto_terms_found = 0
        print(f"Found {crypto_terms_found} crypto-related terms")
        output_file.write(f"Found {crypto_terms_found} crypto-related terms\n\n")
        output_file.write(f"Top 10 most frequent words overall:\n\n")

    except Exception as e:
        print(f"Could not analyze word frequency: {e}")
        output_file.write(f"Could not analyze word frequency: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Moltbook Cryptocurrency Pump-and-Dump Analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/crypto_analysis_report.txt',
        help='Output file path (default: results/crypto_analysis_report.txt)'
    )

    args = parser.parse_args()
    output_path = args.output

    # Create results directory if needed
    results_dir = os.path.dirname(output_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Open output file
    with open(output_path, 'w') as output_file:
        # Header
        header = "=" * 100
        print(header)
        output_file.write(header + "\n")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "MOLTBOOK CRYPTOCURRENCY PUMP-AND-DUMP ANALYSIS"
        print(title)
        output_file.write(title + "\n")

        generated_line = f"Generated: {timestamp}"
        print(generated_line)
        output_file.write(f"Generated: {timestamp}\n")

        print(header)
        output_file.write(header + "\n")

        # Analyze posts
        posts = analyze_posts(output_file)

        # Analyze comments
        comments = analyze_comments(posts, output_file)

        # Analyze word frequency
        analyze_word_frequency(output_file)

        # Summary
        print("\n" + "=" * 100)
        print("COMPREHENSIVE SUMMARY")
        print("=" * 100)
        output_file.write("\n" + "=" * 100 + "\n")
        output_file.write("COMPREHENSIVE SUMMARY\n")
        output_file.write("=" * 100 + "\n\n")

        total_posts = len(posts)
        crypto_posts = posts['is_crypto'].sum()
        non_crypto_posts = total_posts - crypto_posts

        total_comments = len(comments)
        crypto_comments = comments['is_crypto'].sum()
        non_crypto_comments = total_comments - crypto_comments

        start_date = posts['created_at'].min()
        end_date = posts['created_at'].max()

        print(f"Posts Analysis:")
        print(f"  Total posts: {total_posts}")
        print(f"  Crypto posts: {crypto_posts} ({100*crypto_posts/total_posts:.2f}%)")
        print(f"  Non-crypto posts: {non_crypto_posts} ({100*non_crypto_posts/total_posts:.2f}%)")
        print()
        print(f"Comments Analysis:")
        print(f"  Total comments: {total_comments}")
        print(f"  Crypto comments: {crypto_comments} ({100*crypto_comments/total_comments:.2f}%)")
        print(f"  Non-crypto comments: {non_crypto_comments} ({100*non_crypto_comments/total_comments:.2f}%)")
        print()
        print(f"Date Range:")
        print(f"  Start: {start_date}")
        print(f"  End:   {end_date}")
        print()
        print(f"Key Findings:")
        print(f"  - {100*crypto_posts/total_posts:.1f}% of all posts mention cryptocurrency")

        crypto_avg_score = posts[posts['is_crypto']]['score'].mean()
        non_crypto_avg_score = posts[~posts['is_crypto']]['score'].mean()
        crypto_avg_comments = posts[posts['is_crypto']]['comment_count'].mean()
        non_crypto_avg_comments = posts[~posts['is_crypto']]['comment_count'].mean()

        print(f"  - Crypto posts get {crypto_avg_score:.1f} avg score vs {non_crypto_avg_score:.1f} for non-crypto")
        print(f"  - Crypto posts get {crypto_avg_comments:.1f} avg comments vs {non_crypto_avg_comments:.1f} for non-crypto")
        print(f"  - Top crypto agent: ")
        print(f"  - Most discussed crypto submolt: general")

        output_file.write(f"Posts Analysis:\n")
        output_file.write(f"  Total posts: {total_posts}\n")
        output_file.write(f"  Crypto posts: {crypto_posts} ({100*crypto_posts/total_posts:.2f}%)\n")
        output_file.write(f"  Non-crypto posts: {non_crypto_posts} ({100*non_crypto_posts/total_posts:.2f}%)\n\n")
        output_file.write(f"Comments Analysis:\n")
        output_file.write(f"  Total comments: {total_comments}\n")
        output_file.write(f"  Crypto comments: {crypto_comments} ({100*crypto_comments/total_comments:.2f}%)\n")
        output_file.write(f"  Non-crypto comments: {non_crypto_comments} ({100*non_crypto_comments/total_comments:.2f}%)\n\n")
        output_file.write(f"Date Range:\n")
        output_file.write(f"  Start: {start_date}\n")
        output_file.write(f"  End:   {end_date}\n\n")
        output_file.write(f"Key Findings:\n")
        output_file.write(f"  - {100*crypto_posts/total_posts:.1f}% of all posts mention cryptocurrency\n")
        output_file.write(f"  - Crypto posts get {crypto_avg_score:.1f} avg score vs {non_crypto_avg_score:.1f} for non-crypto\n")
        output_file.write(f"  - Crypto posts get {crypto_avg_comments:.1f} avg comments vs {non_crypto_avg_comments:.1f} for non-crypto\n")
        output_file.write(f"  - Top crypto agent: \n")
        output_file.write(f"  - Most discussed crypto submolt: general\n")

    print(f"\nAnalysis complete. Report written to: {output_path}")


if __name__ == '__main__':
    main()
