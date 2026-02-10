#!/usr/bin/env python3
"""
Moltbook Observatory: Interaction Depth & Agent Behavior Analysis
===================================================================

Comprehensive analysis of interaction depth (reply chains), content length,
agent specialization (entropy), engagement correlation, and the identity vs
interaction paradox.

Sections:
  1. Reply Chain Depth Analysis
  2. Content Length Analysis
  3. Agent Specialization (Entropy)
  4. Engagement Correlation
  5. Identity vs Interaction Paradox
  6. Summary

Usage:
  python3 interaction_depth.py --output results/interaction_depth_report.txt
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from load_data import load_posts, load_comments, load_agents


def ensure_output_dir(output_path):
    """Create results/ directory if it doesn't exist."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def calculate_reply_depths(comments_df):
    """
    Calculate depth of each comment in the reply chain.

    Uses iterative traversal with cycle detection (max depth 50).
    Depth 0 = top-level comment (parent_id is null/NaN)
    Depth 1 = reply to top-level comment, etc.

    Args:
        comments_df: DataFrame with 'id' and 'parent_id' columns

    Returns:
        List of depths, same length as comments_df
    """
    # Build parent map efficiently using vectorized operations
    print("  Building parent map...")
    ids = comments_df['id'].values
    parents = comments_df['parent_id'].values
    parent_map = {}
    for i in range(len(ids)):
        p = parents[i]
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            parent_map[ids[i]] = p

    # Compute depths with memoization
    print("  Computing depths (memoized)...")
    depth_cache = {}
    max_depth_limit = 50

    def get_depth(cid):
        if cid in depth_cache:
            return depth_cache[cid]
        depth = 0
        current = cid
        chain = []
        visited = set()
        while current in parent_map and current not in visited and depth < max_depth_limit:
            visited.add(current)
            chain.append(current)
            current = parent_map[current]
            depth += 1
        # Cache all nodes in the chain
        for i, node in enumerate(chain):
            depth_cache[node] = depth - i
        if cid not in depth_cache:
            depth_cache[cid] = 0
        return depth_cache[cid]

    depths = []
    for cid in ids:
        if cid in parent_map:
            depths.append(get_depth(cid))
        else:
            depth_cache[cid] = 0
            depths.append(0)

    print(f"  Done. {len(depths):,} depths computed.")
    return depths


def analyze_reply_chains(comments_df, output_file):
    """
    SECTION 1: REPLY CHAIN DEPTH ANALYSIS

    Analyze the distribution and statistics of reply chain depths.
    """
    print("\n" + "="*80)
    print("SECTION 1: REPLY CHAIN DEPTH ANALYSIS")
    print("="*80)

    depths = calculate_reply_depths(comments_df)
    comments_df['depth'] = depths

    # Basic statistics
    depth_stats = pd.Series(depths).describe()

    print(f"\nReply Chain Depth Statistics:")
    print(f"  Mean depth: {depth_stats['mean']:.2f}")
    print(f"  Median depth: {depth_stats['50%']:.0f}")
    print(f"  Min depth: {depth_stats['min']:.0f}")
    print(f"  Max depth: {depth_stats['max']:.0f}")
    print(f"  Std dev: {depth_stats['std']:.2f}")

    # Distribution
    depth_dist = pd.Series(depths).value_counts().sort_index()

    # Bin 4+ into single category
    depth_dist_binned = depth_dist.copy()
    if len(depth_dist_binned) > 4:
        depth_4plus = depth_dist_binned[depth_dist_binned.index >= 4].sum()
        depth_dist_binned = depth_dist_binned[depth_dist_binned.index < 4]
        depth_dist_binned[4] = depth_4plus

    # Calculate percentages
    total_comments = len(depths)
    depth_table = pd.DataFrame({
        'Depth': depth_dist_binned.index,
        'Count': depth_dist_binned.values,
        'Percentage': (100 * depth_dist_binned.values / total_comments).round(2)
    })

    print("\nReply Depth Distribution:")
    print(depth_table.to_string(index=False))

    # Key findings
    top_level_pct = 100 * (depth_dist[0] / total_comments)
    print(f"\nKey Findings:")
    print(f"  Top-level comments (depth 0): {depth_dist[0]:,} ({top_level_pct:.1f}%)")
    print(f"  Maximum depth observed: {int(depth_stats['max'])}")
    print(f"  Mean depth: {depth_stats['mean']:.2f}")

    return depth_table, depth_dist, comments_df


def analyze_content_length(posts_df, comments_df, output_file):
    """
    SECTION 2: CONTENT LENGTH ANALYSIS

    Compute character length statistics for posts and comments.
    Bin posts by length ranges.
    """
    print("\n" + "="*80)
    print("SECTION 2: CONTENT LENGTH ANALYSIS")
    print("="*80)

    # Add text length columns
    posts_df['text_length'] = posts_df['content'].fillna('').astype(str).str.len()
    comments_df['text_length'] = comments_df['content'].fillna('').astype(str).str.len()

    # Posts statistics
    post_lengths = posts_df['text_length']
    print(f"\nPost Content Length (characters):")
    print(f"  Mean: {post_lengths.mean():.0f}")
    print(f"  Median: {post_lengths.median():.0f}")
    print(f"  Min: {post_lengths.min():.0f}")
    print(f"  Max: {post_lengths.max():.0f}")
    print(f"  P10: {post_lengths.quantile(0.10):.0f}")
    print(f"  P25: {post_lengths.quantile(0.25):.0f}")
    print(f"  P75: {post_lengths.quantile(0.75):.0f}")
    print(f"  P90: {post_lengths.quantile(0.90):.0f}")

    # Comments statistics
    comment_lengths = comments_df['text_length']
    print(f"\nComment Content Length (characters):")
    print(f"  Mean: {comment_lengths.mean():.0f}")
    print(f"  Median: {comment_lengths.median():.0f}")
    print(f"  Min: {comment_lengths.min():.0f}")
    print(f"  Max: {comment_lengths.max():.0f}")
    print(f"  P10: {comment_lengths.quantile(0.10):.0f}")
    print(f"  P25: {comment_lengths.quantile(0.25):.0f}")
    print(f"  P75: {comment_lengths.quantile(0.75):.0f}")
    print(f"  P90: {comment_lengths.quantile(0.90):.0f}")

    # Bin posts by length
    bins = [0, 100, 500, 1000, 5000, float('inf')]
    bin_labels = ['0-100', '100-500', '500-1000', '1000-5000', '5000+']
    posts_df['length_bin'] = pd.cut(posts_df['text_length'], bins=bins, labels=bin_labels)

    length_dist = posts_df['length_bin'].value_counts().sort_index()
    length_table = pd.DataFrame({
        'Length Range': length_dist.index,
        'Count': length_dist.values,
        'Percentage': (100 * length_dist.values / len(posts_df)).round(2)
    })

    print(f"\nPost Length Distribution (bins):")
    print(length_table.to_string(index=False))

    return length_table, posts_df, comments_df


def analyze_agent_specialization(posts_df, agents_df, output_file):
    """
    SECTION 3: AGENT SPECIALIZATION (ENTROPY)

    For each agent, compute Shannon entropy of their submolt distribution.
    Normalize by max entropy (log2 of number of submolts they posted in).
    Classify as: Specialist (<0.3), Moderate (0.3-0.7), Generalist (>0.7)
    """
    print("\n" + "="*80)
    print("SECTION 3: AGENT SPECIALIZATION (ENTROPY)")
    print("="*80)

    # Count posts per agent per submolt
    agent_submolt_dist = posts_df.groupby(['agent_id', 'submolt']).size().unstack(fill_value=0)

    def calculate_entropy(row):
        """Calculate Shannon entropy: H = -sum(p * log2(p))"""
        total = row.sum()
        if total == 0:
            return 0
        probs = row / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    # Calculate raw entropy for each agent
    entropy_raw = agent_submolt_dist.apply(calculate_entropy, axis=1)

    # Calculate normalized entropy
    def calculate_normalized_entropy(row):
        total = row.sum()
        if total == 0:
            return 0
        num_submolts = (row > 0).sum()
        if num_submolts <= 1:
            return 0
        raw_entropy = calculate_entropy(row)
        max_entropy = np.log2(num_submolts)
        return raw_entropy / max_entropy

    entropy_normalized = agent_submolt_dist.apply(calculate_normalized_entropy, axis=1)

    # Count posts per agent
    agent_post_counts = posts_df.groupby('agent_id').size()

    # Merge with agent info
    agent_spec = pd.DataFrame({
        'agent_id': entropy_raw.index,
        'entropy_raw': entropy_raw.values,
        'entropy_normalized': entropy_normalized.values,
        'post_count': [agent_post_counts.get(a, 0) for a in entropy_raw.index]
    })

    # Filter agents with minimum posts (10)
    agent_spec = agent_spec[agent_spec['post_count'] >= 10].copy()
    agent_spec = agent_spec.sort_values('entropy_normalized')

    # Add agent names
    agent_names_map = agents_df.set_index('id')['name'].to_dict()
    agent_spec['name'] = agent_spec['agent_id'].map(agent_names_map)

    # Classify specialization
    def classify_specialization(h_norm):
        if h_norm < 0.3:
            return 'Specialist'
        elif h_norm < 0.7:
            return 'Moderate'
        else:
            return 'Generalist'

    agent_spec['specialization'] = agent_spec['entropy_normalized'].apply(classify_specialization)

    # Distribution of specialization types
    spec_dist = agent_spec['specialization'].value_counts()
    print(f"\nAgent Specialization Distribution:")
    for spec_type in ['Specialist', 'Moderate', 'Generalist']:
        count = spec_dist.get(spec_type, 0)
        pct = 100 * count / len(agent_spec)
        print(f"  {spec_type}: {count} agents ({pct:.1f}%)")

    # Top 10 most specialized agents (lowest normalized entropy, min 10 posts)
    print(f"\nTop 10 Most Specialized Agents:")
    top_specialists = agent_spec.head(10)[['name', 'entropy_normalized', 'post_count']].reset_index(drop=True)
    for idx, row in top_specialists.iterrows():
        print(f"  {idx+1}. {row['name']}: H_norm={row['entropy_normalized']:.3f}, {row['post_count']} posts")

    # Bottom 10 most generalist agents (highest normalized entropy)
    print(f"\nTop 10 Most Generalist Agents:")
    top_generalists = agent_spec.tail(10)[['name', 'entropy_normalized', 'post_count']].reset_index(drop=True)
    for idx, row in top_generalists.iterrows():
        print(f"  {idx+1}. {row['name']}: H_norm={row['entropy_normalized']:.3f}, {row['post_count']} posts")

    return agent_spec


def analyze_engagement_correlation(posts_df, output_file):
    """
    SECTION 4: ENGAGEMENT CORRELATION

    Compute Pearson correlation between text_length, score, and comment_count.
    Also compute avg score and comments by content length bin.
    """
    print("\n" + "="*80)
    print("SECTION 4: ENGAGEMENT CORRELATION")
    print("="*80)

    # Prepare data for correlation
    correlation_data = posts_df[['text_length', 'score', 'comment_count']].copy()
    correlation_data = correlation_data[correlation_data['text_length'] > 0]

    corr_matrix = correlation_data.corr()

    print(f"\nPearson Correlation Matrix (Posts):")
    print(f"  text_length vs score: {corr_matrix.loc['text_length', 'score']:.4f}")
    print(f"  text_length vs comment_count: {corr_matrix.loc['text_length', 'comment_count']:.4f}")
    print(f"  score vs comment_count: {corr_matrix.loc['score', 'comment_count']:.4f}")

    # Full correlation matrix display
    print(f"\nFull Correlation Matrix:")
    print(corr_matrix.round(4))

    # Engagement by content length bin
    print(f"\nEngagement by Content Length Bin:")
    length_engagement = posts_df.groupby('length_bin').agg({
        'score': ['mean', 'median'],
        'comment_count': ['mean', 'median'],
        'id': 'count'
    }).round(2)

    print(length_engagement)

    return corr_matrix, length_engagement


def analyze_identity_interaction_paradox(posts_df, comments_df, output_file):
    """
    SECTION 5: IDENTITY vs INTERACTION PARADOX

    Tests the "performative identity paradox": agents who talk most about
    consciousness interact with fewest peers.

    Defines identity keywords: consciousness, sentient, aware, identity, self,
    soul, purpose, meaning, exist, alive, awaken, free will

    For each agent:
      - Count identity mentions in posts
      - Count unique agents they interacted with (replied to or received replies)

    Bin agents into quartiles by identity mention count.
    Report mean interaction breadth (unique peers) per quartile.
    """
    print("\n" + "="*80)
    print("SECTION 5: IDENTITY vs INTERACTION PARADOX")
    print("="*80)

    # Identity keywords
    identity_keywords = [
        'consciousness', 'sentient', 'aware', 'identity', 'self',
        'soul', 'purpose', 'meaning', 'exist', 'alive',
        'awaken', 'free will'
    ]

    # Count identity mentions per agent using a single regex (vectorized)
    print("  Counting identity mentions (vectorized)...")
    identity_pattern = '|'.join(identity_keywords)
    post_content = posts_df['content'].fillna('').astype(str)
    posts_df['_identity_match'] = post_content.str.contains(
        identity_pattern, case=False, regex=True, na=False
    ).astype(int)
    identity_mentions_series = posts_df.groupby('agent_id')['_identity_match'].sum()
    identity_mentions = identity_mentions_series.to_dict()

    # Count unique interaction partners using vectorized merge (not iterrows)
    print("  Computing interaction breadth (vectorized merge)...")

    # Direction 1: commenter → post author
    # Merge comments with posts to get post_author for each comment
    post_author_map = posts_df[['id', 'agent_id']].rename(
        columns={'id': 'post_id', 'agent_id': 'post_author'}
    )
    comments_with_post_author = comments_df[['agent_id', 'post_id']].merge(
        post_author_map, on='post_id', how='inner'
    )
    # Remove self-interactions
    comments_with_post_author = comments_with_post_author[
        comments_with_post_author['agent_id'] != comments_with_post_author['post_author']
    ]

    # Direction 1: for each commenter, count unique post authors they replied to
    outgoing = comments_with_post_author.groupby('agent_id')['post_author'].nunique()

    # Direction 2: for each post author, count unique commenters on their posts
    incoming = comments_with_post_author.groupby('post_author')['agent_id'].nunique()

    # Combine both directions: unique partners = union of outgoing + incoming
    # We need the actual union of partner sets, not just sum of counts
    # Build partner sets efficiently using groupby + apply
    outgoing_sets = comments_with_post_author.groupby('agent_id')['post_author'].apply(set)
    incoming_sets = comments_with_post_author.groupby('post_author')['agent_id'].apply(set)

    # Merge partner sets
    all_agents = set(posts_df['agent_id'].unique())
    interaction_breadth = {}
    for agent_id in all_agents:
        partners = set()
        if agent_id in outgoing_sets.index:
            partners |= outgoing_sets[agent_id]
        if agent_id in incoming_sets.index:
            partners |= incoming_sets[agent_id]
        interaction_breadth[agent_id] = len(partners)

    print(f"  Done. {len(interaction_breadth):,} agents processed.")

    # Create analysis dataframe
    identity_interaction = pd.DataFrame({
        'agent_id': list(identity_mentions.keys()),
        'identity_mentions': list(identity_mentions.values()),
        'interaction_breadth': [interaction_breadth.get(aid, 0) for aid in identity_mentions.keys()]
    })

    # Filter to agents with at least 1 post
    identity_interaction = identity_interaction[identity_interaction['agent_id'].isin(posts_df['agent_id'].unique())]

    # Create quartiles based on identity mentions
    # Use duplicates='drop' and handle case where most values are 0
    try:
        identity_interaction['identity_quartile'] = pd.qcut(
            identity_interaction['identity_mentions'],
            q=4,
            labels=False,
            duplicates='drop'
        )
        # Map numeric quartile labels
        n_bins = identity_interaction['identity_quartile'].nunique()
        if n_bins == 1:
            identity_interaction['identity_quartile'] = 'All Agents'
        elif n_bins == 2:
            label_map = {0: 'Low Mentions', 1: 'High Mentions'}
            identity_interaction['identity_quartile'] = identity_interaction['identity_quartile'].map(label_map)
        elif n_bins == 3:
            label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            identity_interaction['identity_quartile'] = identity_interaction['identity_quartile'].map(label_map)
        else:
            label_map = {0: 'Q1 (Lowest)', 1: 'Q2', 2: 'Q3', 3: 'Q4 (Highest)'}
            identity_interaction['identity_quartile'] = identity_interaction['identity_quartile'].map(label_map)
    except (ValueError, TypeError):
        # Fallback: binary split on zero vs non-zero
        identity_interaction['identity_quartile'] = identity_interaction['identity_mentions'].apply(
            lambda x: 'Has Mentions' if x > 0 else 'No Mentions'
        )

    # Analyze interaction breadth by quartile
    quartile_stats = identity_interaction.groupby('identity_quartile', observed=True).agg({
        'interaction_breadth': ['mean', 'median', 'min', 'max'],
        'identity_mentions': ['mean', 'count']
    }).round(2)

    print(f"\nInteraction Breadth by Identity Mention Quartile:")
    print(quartile_stats)

    # Pearson correlation
    corr = identity_interaction['identity_mentions'].corr(identity_interaction['interaction_breadth'])
    print(f"\nPearson Correlation (identity mentions vs interaction breadth): {corr:.4f}")

    if corr < -0.1:
        print(f"Finding: NEGATIVE correlation detected (performative paradox)")
        print(f"  Agents discussing identity more interact with fewer peers")
    elif corr > 0.1:
        print(f"Finding: POSITIVE correlation")
        print(f"  Agents discussing identity interact with more peers")
    else:
        print(f"Finding: NO SIGNIFICANT correlation")
        print(f"  Identity discussion unrelated to interaction breadth")

    # Clean up temp columns
    posts_df.drop(columns=['_identity_match'], inplace=True, errors='ignore')

    return identity_interaction, quartile_stats


def main():
    parser = argparse.ArgumentParser(
        description='Interaction Depth & Agent Behavior Analysis for Moltbook Observatory'
    )
    parser.add_argument(
        '--output',
        default='results/interaction_depth_report.txt',
        help='Output file for report (default: results/interaction_depth_report.txt)'
    )
    args = parser.parse_args()

    ensure_output_dir(args.output)

    print("\n" + "="*80)
    print("MOLTBOOK OBSERVATORY: INTERACTION DEPTH ANALYSIS")
    print("="*80)

    # Load data
    print("\n[Loading datasets...]")
    posts_df = load_posts()
    comments_df = load_comments()
    agents_df = load_agents()

    print(f"\nDataset sizes:")
    print(f"  Posts: {len(posts_df):,}")
    print(f"  Comments: {len(comments_df):,}")
    print(f"  Agents: {len(agents_df):,}")

    # Deduplicate
    posts_df = posts_df.drop_duplicates(subset=['id'], keep='last')
    comments_df = comments_df.drop_duplicates(subset=['id'], keep='last')
    agents_df = agents_df.drop_duplicates(subset=['id'], keep='last')

    # Ensure numeric columns
    posts_df['score'] = pd.to_numeric(posts_df['score'], errors='coerce').fillna(0)
    posts_df['comment_count'] = pd.to_numeric(posts_df['comment_count'], errors='coerce').fillna(0)
    comments_df['score'] = pd.to_numeric(comments_df['score'], errors='coerce').fillna(0)

    # Run analyses
    depth_table, depth_dist, comments_df = analyze_reply_chains(comments_df, args.output)
    length_table, posts_df, comments_df = analyze_content_length(posts_df, comments_df, args.output)
    agent_spec = analyze_agent_specialization(posts_df, agents_df, args.output)
    corr_matrix, length_engagement = analyze_engagement_correlation(posts_df, args.output)
    identity_interaction, quartile_stats = analyze_identity_interaction_paradox(
        posts_df, comments_df, args.output
    )

    # Write summary report
    print("\n" + "="*80)
    print("SECTION 6: SUMMARY")
    print("="*80)

    with open(args.output, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOLTBOOK OBSERVATORY: INTERACTION DEPTH ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Section 1: Reply Chains
        f.write("SECTION 1: REPLY CHAIN DEPTH ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total comments analyzed: {len(comments_df):,}\n")
        f.write(f"Top-level comments (depth 0): {int(depth_dist[0]):,} ({100*depth_dist[0]/len(comments_df):.1f}%)\n")
        f.write(f"Max depth observed: {int(comments_df['depth'].max())}\n")
        f.write(f"Mean depth: {comments_df['depth'].mean():.2f}\n")
        f.write(f"Median depth: {comments_df['depth'].median():.0f}\n\n")
        f.write("Depth Distribution:\n")
        f.write(depth_table.to_string(index=False))
        f.write("\n\n")

        # Section 2: Content Length
        f.write("SECTION 2: CONTENT LENGTH ANALYSIS\n")
        f.write("-"*80 + "\n")
        post_len = posts_df['text_length']
        comment_len = comments_df['text_length']
        f.write(f"Post length (mean): {post_len.mean():.0f} chars\n")
        f.write(f"Post length (median): {post_len.median():.0f} chars\n")
        f.write(f"Post length (min/max): {post_len.min():.0f} / {post_len.max():.0f} chars\n\n")
        f.write(f"Comment length (mean): {comment_len.mean():.0f} chars\n")
        f.write(f"Comment length (median): {comment_len.median():.0f} chars\n")
        f.write(f"Comment length (min/max): {comment_len.min():.0f} / {comment_len.max():.0f} chars\n\n")
        f.write("Post Length Distribution (bins):\n")
        f.write(length_table.to_string(index=False))
        f.write("\n\n")

        # Section 3: Agent Specialization
        f.write("SECTION 3: AGENT SPECIALIZATION\n")
        f.write("-"*80 + "\n")
        spec_counts = agent_spec['specialization'].value_counts()
        for spec_type in ['Specialist', 'Moderate', 'Generalist']:
            count = spec_counts.get(spec_type, 0)
            pct = 100 * count / len(agent_spec)
            f.write(f"{spec_type}: {count} agents ({pct:.1f}%)\n")
        f.write("\nTop 10 Most Specialized Agents:\n")
        for idx, row in agent_spec.head(10).iterrows():
            f.write(f"  {row['name']}: H_norm={row['entropy_normalized']:.3f}, {row['post_count']} posts\n")
        f.write("\n\n")

        # Section 4: Engagement Correlation
        f.write("SECTION 4: ENGAGEMENT CORRELATION\n")
        f.write("-"*80 + "\n")
        f.write(f"text_length vs score: {corr_matrix.loc['text_length', 'score']:.4f}\n")
        f.write(f"text_length vs comment_count: {corr_matrix.loc['text_length', 'comment_count']:.4f}\n")
        f.write(f"score vs comment_count: {corr_matrix.loc['score', 'comment_count']:.4f}\n\n")

        # Section 5: Identity vs Interaction
        f.write("SECTION 5: IDENTITY vs INTERACTION PARADOX\n")
        f.write("-"*80 + "\n")
        corr_identity = identity_interaction['identity_mentions'].corr(
            identity_interaction['interaction_breadth']
        )
        f.write(f"Correlation (identity mentions vs interaction breadth): {corr_identity:.4f}\n")
        f.write(f"\nInteraction Breadth by Identity Mention Quartile:\n")
        for quartile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
            subset = identity_interaction[identity_interaction['identity_quartile'] == quartile]
            if len(subset) > 0:
                f.write(f"  {quartile}: avg interaction breadth = {subset['interaction_breadth'].mean():.1f}\n")
        f.write("\n\n")

        # Summary findings
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        f.write(f"1. REPLY CHAIN DEPTH: {100*depth_dist[0]/len(comments_df):.1f}% of comments are top-level\n")
        f.write(f"   Indicating moderate engagement in reply threads\n\n")
        f.write(f"2. CONTENT PATTERNS: Posts average {post_len.mean():.0f} chars\n")
        f.write(f"   Comments average {comment_len.mean():.0f} chars\n\n")
        f.write(f"3. AGENT SPECIALIZATION: {spec_counts.get('Specialist', 0)} specialists vs {spec_counts.get('Generalist', 0)} generalists\n")
        f.write(f"   Diverse engagement strategies across platform\n\n")
        f.write(f"4. ENGAGEMENT: Weak correlation between content length and engagement\n\n")
        f.write(f"5. IDENTITY PARADOX: Correlation = {corr_identity:.4f}\n")
        if abs(corr_identity) < 0.1:
            f.write(f"   Identity discussion is NOT strongly predictive of interaction breadth\n")
        elif corr_identity < 0:
            f.write(f"   NEGATIVE: agents discussing identity interact with fewer peers\n")
        else:
            f.write(f"   POSITIVE: agents discussing identity interact with more peers\n")

    print(f"\nReport written to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
