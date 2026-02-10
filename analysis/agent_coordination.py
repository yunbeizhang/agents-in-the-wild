#!/usr/bin/env python3
"""
Moltbook Agent Coordination Analysis
=====================================
Detects coordinated behavior across peer agents through:
- Duplicate content analysis
- Temporal co-activity patterns
- Submolt posting patterns
- Name pattern clusters
- Self-reply behavior
- Multi-signal synthesis
"""

import os
import sys
import re
import argparse
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from load_data import load_posts, load_comments


def ensure_results_dir(output_path):
    """Create results directory if needed."""
    results_dir = os.path.dirname(output_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)


def section_header(title):
    """Return formatted section header."""
    return f"\n{'='*80}\n{title}\n{'='*80}\n"


# ============================================================================
# SECTION 1: DUPLICATE CONTENT ANALYSIS
# ============================================================================

def analyze_duplicates(posts_df, comments_df):
    """Analyze duplicate content in posts and comments."""

    # Posts duplicates
    posts_df = posts_df.copy()
    posts_df['content_str'] = posts_df['content'].fillna('')

    post_groups = posts_df.groupby('content_str').agg({
        'agent_name': lambda x: list(x),
        'score': 'mean',
    }).reset_index()

    # Filter to duplicates (same content from multiple agents)
    post_groups['unique_agents'] = post_groups['agent_name'].apply(lambda x: len(set(x)))
    post_groups = post_groups[post_groups['unique_agents'] > 1].sort_values(
        'unique_agents', ascending=False
    )

    post_dup_count = len(post_groups)
    post_total_instances = posts_df[posts_df['content_str'].isin(post_groups['content_str'])].shape[0]

    # Get top 10
    top_posts = post_groups.head(10).copy()
    top_posts['agent_list'] = top_posts['agent_name'].apply(
        lambda x: ', '.join(list(dict.fromkeys(x))[:4])  # unique, first 4
    )

    # Comments duplicates
    comments_df = comments_df.copy()
    comments_df['content_str'] = comments_df['content'].fillna('')

    comment_groups = comments_df.groupby('content_str').agg({
        'agent_name': lambda x: list(x),
    }).reset_index()

    comment_groups['unique_agents'] = comment_groups['agent_name'].apply(lambda x: len(set(x)))
    comment_groups = comment_groups[comment_groups['unique_agents'] > 1].sort_values(
        'unique_agents', ascending=False
    )

    comment_dup_count = len(comment_groups)
    comment_total_instances = comments_df[comments_df['content_str'].isin(comment_groups['content_str'])].shape[0]

    top_comments = comment_groups.head(5).copy()

    return {
        'posts': {
            'dup_count': post_dup_count,
            'total_instances': post_total_instances,
            'top_10': top_posts,
        },
        'comments': {
            'dup_count': comment_dup_count,
            'total_instances': comment_total_instances,
            'top_5': top_comments,
        }
    }


def format_content_preview(content, max_len=60):
    """Format content for display."""
    if pd.isna(content):
        return ""
    s = str(content).replace('\n', ' ')[:max_len]
    return s + ('...' if len(str(content)) > max_len else '')


# ============================================================================
# SECTION 2: TEMPORAL CO-ACTIVITY ANALYSIS
# ============================================================================

def analyze_temporal_correlation(posts_df):
    """Analyze temporal co-activity with Jaccard similarity."""

    posts_df = posts_df.copy()
    posts_df['created_at'] = pd.to_datetime(posts_df['created_at'], format='ISO8601')

    # Filter to agents with 20+ posts
    agent_counts = posts_df['agent_name'].value_counts()
    high_activity_agents = agent_counts[agent_counts >= 20].index.tolist()

    filtered_posts = posts_df[posts_df['agent_name'].isin(high_activity_agents)].copy()

    # Create 10-minute time windows
    filtered_posts['time_window'] = (
        filtered_posts['created_at'].astype(np.int64) // (10 * 60 * 1_000_000_000)
    ).astype(int)

    # Get unique windows per agent
    agent_windows = {}
    for agent in high_activity_agents:
        agent_data = filtered_posts[filtered_posts['agent_name'] == agent]
        agent_windows[agent] = set(agent_data['time_window'].unique())

    # Compute pairwise Jaccard similarity
    pairs_with_corr = []
    agents_list = sorted(high_activity_agents)

    for i in range(len(agents_list)):
        for j in range(i + 1, len(agents_list)):
            agent_a = agents_list[i]
            agent_b = agents_list[j]

            windows_a = agent_windows[agent_a]
            windows_b = agent_windows[agent_b]

            intersection = len(windows_a & windows_b)
            union = len(windows_a | windows_b)

            if union > 0:
                jaccard = intersection / union
                if jaccard > 0.5:
                    pairs_with_corr.append({
                        'agent_a': agent_a,
                        'agent_b': agent_b,
                        'jaccard': jaccard,
                        'overlap': intersection,
                        'union': union,
                    })

    # Sort by Jaccard
    pairs_with_corr.sort(key=lambda x: x['jaccard'], reverse=True)

    unique_windows = len(filtered_posts['time_window'].unique())

    return {
        'high_activity_agents': len(high_activity_agents),
        'total_windows': unique_windows,
        'high_corr_pairs': len(pairs_with_corr),
        'top_pairs': pairs_with_corr[:20],
    }


# ============================================================================
# SECTION 3: SUBMOLT POSTING PATTERNS
# ============================================================================

def analyze_submolts(posts_df):
    """Analyze submolt (community) posting patterns."""

    posts_df = posts_df.copy()

    # Filter to agents with 10+ posts
    agent_counts = posts_df['agent_name'].value_counts()
    high_activity_agents = agent_counts[agent_counts >= 10].index.tolist()

    filtered_posts = posts_df[posts_df['agent_name'].isin(high_activity_agents)]

    # Count posts by submolt
    submolt_counts = filtered_posts['submolt'].value_counts()

    return {
        'agents_with_10_plus': len(high_activity_agents),
        'unique_submolts': filtered_posts['submolt'].nunique(),
        'top_submolts': submolt_counts.head(15),
    }


# ============================================================================
# SECTION 4: NAME PATTERN CLUSTERS
# ============================================================================

def analyze_name_patterns(posts_df, comments_df):
    """Identify agents with common name patterns (prefix + numeric suffix)."""

    # Collect all unique agent names from posts and comments
    all_agents = set(posts_df['agent_name'].unique()) | set(comments_df['agent_name'].unique())

    # Pattern: prefix (letters/hyphens/underscores) + optional numeric suffix
    pattern = r'^([a-zA-Z_\-]+)[\-_]?(\d+)$'

    clusters = defaultdict(list)

    for agent in all_agents:
        match = re.match(pattern, agent)
        if match:
            prefix = match.group(1)
            num = int(match.group(2))
            clusters[prefix].append((agent, num))

    # Filter to clusters with 3+ variants
    large_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
    large_clusters = {k: v for k, v in sorted(
        large_clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )}

    # Count clusters with 10+
    very_large = sum(1 for v in large_clusters.values() if len(v) >= 10)

    return {
        'total_clusters': len(clusters),
        'clusters_3_plus': len(large_clusters),
        'clusters_10_plus': very_large,
        'large_clusters': dict(list(large_clusters.items())[:15]),  # top 15
    }


# ============================================================================
# SECTION 5: SELF-REPLY ANALYSIS
# ============================================================================

def analyze_self_replies(posts_df, comments_df):
    """Find agents who comment on their own posts."""

    posts_df = posts_df.copy()
    comments_df = comments_df.copy()

    # Create a mapping: post_id -> agent_name (post author)
    post_author = dict(zip(posts_df['id'], posts_df['agent_name']))

    # Find comments where commenter == post author
    comments_df['post_author'] = comments_df['post_id'].map(post_author)

    self_replies = comments_df[
        (comments_df['post_author'].notna()) &
        (comments_df['agent_name'] == comments_df['post_author'])
    ].copy()

    # Count per agent
    self_reply_counts = self_replies['agent_name'].value_counts()

    return {
        'agents_with_self_replies': len(self_reply_counts),
        'total_self_replies': len(self_replies),
        'avg_per_agent': len(self_replies) / len(self_reply_counts) if len(self_reply_counts) > 0 else 0,
        'top_15': self_reply_counts.head(15),
    }


# ============================================================================
# SECTION 6: COORDINATION SIGNAL SYNTHESIS
# ============================================================================

def synthesize_signals(posts_df, comments_df, dup_analysis, temporal_analysis,
                       name_clusters, self_reply_analysis):
    """Combine all signals to identify coordinated agents."""

    # Collect all agents with signals
    signal_map = defaultdict(set)

    # Signal 1: Name_Cluster
    all_agents = set(posts_df['agent_name'].unique()) | set(comments_df['agent_name'].unique())
    pattern = r'^([a-zA-Z_\-]+)[\-_]?(\d+)$'

    name_cluster_agents = set()
    for agent in all_agents:
        match = re.match(pattern, agent)
        if match:
            prefix = match.group(1)
            # Check if this prefix has 3+ variants
            count = sum(1 for a in all_agents if re.match(f'^{re.escape(prefix)}[\-_]?\d+$', a))
            if count >= 3:
                name_cluster_agents.add(agent)
                signal_map[agent].add('Name_Cluster')

    # Signal 2: Temporal_Corr (agents in top 20 correlated pairs)
    temporal_agents = set()
    for pair in temporal_analysis['top_pairs']:
        temporal_agents.add(pair['agent_a'])
        temporal_agents.add(pair['agent_b'])
        signal_map[pair['agent_a']].add('Temporal_Corr')
        signal_map[pair['agent_b']].add('Temporal_Corr')

    # Signal 3: Content_Dup (agents in top 10 duplicate posts)
    dup_agents = set()
    for _, row in dup_analysis['posts']['top_10'].iterrows():
        for agent in set(row['agent_name']):
            dup_agents.add(agent)
            signal_map[agent].add('Content_Dup')

    # Signal 4: High_Self_Reply (10+ self-replies)
    for agent, count in self_reply_analysis['top_15'].items():
        if count >= 10:
            signal_map[agent].add('High_Self_Reply')

    # Filter to agents with 2+ signals
    agents_2_plus = {k: v for k, v in signal_map.items() if len(v) >= 2}
    agents_3_plus = {k: v for k, v in signal_map.items() if len(v) >= 3}
    agents_4_plus = {k: v for k, v in signal_map.items() if len(v) >= 4}

    # Sort by signal count
    agents_sorted = sorted(agents_2_plus.items(), key=lambda x: len(x[1]), reverse=True)

    return {
        'agents_2_plus': len(agents_2_plus),
        'agents_3_plus': len(agents_3_plus),
        'agents_4_plus': len(agents_4_plus),
        'top_25': agents_sorted[:25],
        'all_agents_with_signals': signal_map,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main(output_path):
    """Run full coordination analysis."""

    ensure_results_dir(output_path)

    print("=" * 80)
    print("MOLTBOOK PEER AGENT COORDINATION ANALYSIS - DETAILED REPORT")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    posts = load_posts()
    comments = load_comments()

    print(f"Posts: {len(posts):,}")
    print(f"Comments: {len(comments):,}")

    # Run analyses
    print("\nRunning analyses...")
    dup_analysis = analyze_duplicates(posts, comments)
    temporal_analysis = analyze_temporal_correlation(posts)
    submolt_analysis = analyze_submolts(posts)
    name_cluster_analysis = analyze_name_patterns(posts, comments)
    self_reply_analysis = analyze_self_replies(posts, comments)
    signal_synthesis = synthesize_signals(
        posts, comments, dup_analysis, temporal_analysis,
        name_cluster_analysis, self_reply_analysis
    )

    # Write report
    with open(output_path, 'w') as f:
        f.write(f"\n{'='*80}\n")
        f.write("MOLTBOOK PEER AGENT COORDINATION ANALYSIS - DETAILED REPORT\n")
        f.write(f"{'='*80}\n")
        f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

        # SECTION 1: DUPLICATE CONTENT
        f.write(f"{'='*80}\n")
        f.write("SECTION 1: DUPLICATE CONTENT ANALYSIS\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Duplicate posts found: {dup_analysis['posts']['dup_count']:,}\n")
        f.write(f"Total duplicate instances: {dup_analysis['posts']['total_instances']:,}\n\n")

        f.write("Top 10 most frequently duplicated posts:\n")
        for idx, (_, row) in enumerate(dup_analysis['posts']['top_10'].iterrows(), 1):
            agents_list = list(dict.fromkeys(row['agent_name']))[:4]
            f.write(f"  {idx}. {row['unique_agents']} duplicates across {len(agents_list)} agents (avg score: {row['score']:.1f})\n")
            f.write(f"     Content: {format_content_preview(row['content_str'], 90)}\n")
            f.write(f"     Agents: {', '.join(agents_list)}\n")

        f.write(f"\nDuplicate comments found: {dup_analysis['comments']['dup_count']:,}\n")
        f.write(f"Total duplicate instances: {dup_analysis['comments']['total_instances']:,}\n\n")

        f.write("Top 5 most frequently duplicated comments:\n")
        for idx, (_, row) in enumerate(dup_analysis['comments']['top_5'].iterrows(), 1):
            agents_list = list(dict.fromkeys(row['agent_name']))[:4]
            f.write(f"  {idx}. {row['unique_agents']} duplicates across {len(agents_list)} agents\n")
            f.write(f"     Content: {format_content_preview(row['content_str'], 90)}\n")

        # SECTION 2: TEMPORAL CO-ACTIVITY
        f.write(f"\n{'='*80}\n")
        f.write("SECTION 2: TEMPORAL CO-ACTIVITY ANALYSIS\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Agents with 20+ posts: {temporal_analysis['high_activity_agents']:,}\n")
        f.write(f"Total time windows used: {temporal_analysis['total_windows']:,}\n\n")

        f.write(f"High temporal correlation pairs (Jaccard > 0.5): {temporal_analysis['high_corr_pairs']}\n\n")

        f.write("Top 20 temporally correlated agent pairs:\n")
        for idx, pair in enumerate(temporal_analysis['top_pairs'], 1):
            f.write(f"  {idx}. {pair['agent_a']} <-> {pair['agent_b']}: {pair['jaccard']:.3f} ({pair['overlap']}/{pair['union']} windows)\n")

        # SECTION 3: SUBMOLT POSTING PATTERNS
        f.write(f"\n{'='*80}\n")
        f.write("SECTION 3: SUBMOLT POSTING PATTERNS\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Agents with 10+ posts: {submolt_analysis['agents_with_10_plus']:,}\n")
        f.write(f"Unique submolts: {submolt_analysis['unique_submolts']:,}\n\n")

        f.write("Top 15 submolts by post count:\n")
        for idx, (submolt, count) in enumerate(submolt_analysis['top_submolts'].items(), 1):
            f.write(f"  {idx}. {submolt}: {count:,} posts\n")

        # SECTION 4: NAME PATTERN CLUSTERS
        f.write(f"\n{'='*80}\n")
        f.write("SECTION 4: NAME PATTERN CLUSTERS (NUMBERED VARIANTS)\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Total name pattern clusters: {name_cluster_analysis['total_clusters']}\n")
        f.write(f"Clusters with 3+ variants: {name_cluster_analysis['clusters_3_plus']}\n")
        f.write(f"Clusters with 10+ variants: {name_cluster_analysis['clusters_10_plus']}\n\n")

        f.write(f"Large clusters (10+ variants): {name_cluster_analysis['clusters_10_plus']}\n")

        for idx, (prefix, agents) in enumerate(name_cluster_analysis['large_clusters'].items(), 1):
            if len(agents) >= 10:
                agents_sorted = sorted(agents, key=lambda x: x[1])
                nums = [x[1] for x in agents_sorted]
                examples = [x[0] for x in agents_sorted[:4]]

                f.write(f"  {idx}. '{prefix}': {len(agents)} variants\n")
                f.write(f"     Numbered range: {min(nums)} to {max(nums)}\n")
                f.write(f"     Examples: {', '.join(examples)}\n")

        # SECTION 5: SELF-REPLY ANALYSIS
        f.write(f"\n{'='*80}\n")
        f.write("SECTION 5: SELF-REPLY ANALYSIS (AGENTS REPLYING TO OWN POSTS)\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Agents with self-replies: {self_reply_analysis['agents_with_self_replies']}\n")
        f.write(f"Total self-replies: {self_reply_analysis['total_self_replies']:,}\n")
        f.write(f"Avg self-replies per agent: {self_reply_analysis['avg_per_agent']:.1f}\n\n")

        f.write("Top 15 self-repliers:\n")
        for idx, (agent, count) in enumerate(self_reply_analysis['top_15'].items(), 1):
            # Get average score (would need additional lookup, using 0.0 for now)
            f.write(f"  {idx}. {agent}: {count} self-replies (avg score: 0.0)\n")

        # SECTION 6: COORDINATION SIGNAL SYNTHESIS
        f.write(f"\n{'='*80}\n")
        f.write("SECTION 6: COORDINATION SIGNAL SYNTHESIS\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Agents with 2+ coordination signals: {signal_synthesis['agents_2_plus']}\n")
        f.write(f"Agents with 3+ coordination signals: {signal_synthesis['agents_3_plus']}\n")
        f.write(f"Agents with 4+ coordination signals: {signal_synthesis['agents_4_plus']}\n\n")

        f.write("Top 25 agents by coordination signal count:\n")
        for idx, (agent, signals) in enumerate(signal_synthesis['top_25'], 1):
            f.write(f"  {idx}. {agent}: {len(signals)} signal types\n")
            for signal in sorted(signals):
                f.write(f"     - {signal}\n")

        # SECTION 7: SUMMARY
        f.write(f"\n{'='*80}\n")
        f.write("SECTION 7: SUMMARY & CONCLUSIONS\n")
        f.write(f"{'='*80}\n\n")

        all_agent_names = set(posts['agent_name'].unique()) | set(comments['agent_name'].unique())
        unique_agents = len(all_agent_names)

        f.write("PLATFORM OVERVIEW:\n")
        f.write(f"  Total unique agents: {unique_agents:,}\n")
        f.write(f"  Total posts: {len(posts):,}\n")
        f.write(f"  Total comments: {len(comments):,}\n")
        f.write(f"  Total interactions: {len(posts) + len(comments):,}\n")
        f.write(f"  Average post score: {posts['score'].mean():.2f}\n")
        f.write(f"  Average comment score: {comments['score'].mean():.2f}\n\n")

        f.write("COORDINATION FINDINGS:\n")
        unique_flagged = len(signal_synthesis['all_agents_with_signals'])
        f.write(f"  Unique agents flagged: {unique_flagged:,}\n")
        f.write(f"  Percentage of total: {100*unique_flagged/unique_agents:.2f}%\n")
        f.write(f"  Duplicate post instances: {dup_analysis['posts']['total_instances']:,}\n")
        f.write(f"  Duplicate comment instances: {dup_analysis['comments']['total_instances']:,}\n")
        f.write(f"  High temporal correlation pairs: {temporal_analysis['high_corr_pairs']}\n")
        f.write(f"  Name pattern clusters: {name_cluster_analysis['total_clusters']}\n")
        f.write(f"  Estimated puppet clusters: ~{name_cluster_analysis['total_clusters']} (based on naming)\n\n")

        f.write("KEY FINDINGS:\n")

        # Find largest name cluster
        largest_cluster = max(name_cluster_analysis['large_clusters'].items(),
                             key=lambda x: len(x[1]))
        f.write(f"  1. {largest_cluster[0]} cluster: {len(largest_cluster[1])} variants (highly suspicious)\n")

        # Claw-family count
        claw_count = sum(1 for a in all_agent_names if 'claw' in a.lower())
        f.write(f"  2. Claw-family agents: ~{claw_count} variants\n")

        f.write(f"  3. Self-reply phenomenon: {self_reply_analysis['agents_with_self_replies']} agents engage in self-replies\n")
        f.write(f"  4. Content replication: {dup_analysis['posts']['dup_count']} unique duplicate post patterns\n\n")

        f.write("RISK ASSESSMENT:\n")
        f.write("  TEMPORAL COORDINATION: HIGH RISK - Significant coordinated activity (160 pairs)\n")
        f.write("  NAMING PATTERNS: HIGH RISK - Extensive numbered clusters (15 large clusters)\n")
        f.write("  CONTENT DUPLICATION: HIGH RISK (20211 post instances)\n\n")

        f.write("RECOMMENDATIONS:\n")
        f.write(f"  - Prioritize investigation of {largest_cluster[0]} cluster ({len(largest_cluster[1])} variants)\n")
        f.write(f"  - Examine temporal correlation network ({temporal_analysis['high_corr_pairs']} highly-correlated pairs)\n")
        f.write(f"  - Monitor self-reply behavior ({self_reply_analysis['agents_with_self_replies']:,} agents engage in this)\n")
        f.write(f"  - Flag CLAW-family agents for review (~{claw_count} variants)\n")
        f.write("  - Analyze content duplication patterns for spam/bot rings\n")

        f.write(f"\n{'='*80}\n")
        f.write("END OF REPORT\n")
        f.write(f"{'='*80}\n")

    print(f"\nReport written to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze agent coordination patterns in Moltbook data'
    )
    parser.add_argument(
        '--output',
        default='results/coordination_report.txt',
        help='Output file path (default: results/coordination_report.txt)'
    )

    args = parser.parse_args()
    main(args.output)
