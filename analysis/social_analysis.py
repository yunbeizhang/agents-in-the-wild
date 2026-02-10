#!/usr/bin/env python3
"""
Moltbook Observatory: Social Phenomena Analysis

Comprehensive analysis of social phenomena in AI agent communities,
including safety discussions, community dynamics, social networks, and
emergent behaviors.

Usage:
    python social_analysis.py --output results/social_analysis_report.txt
"""

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from load_data import (
    load_posts,
    load_comments,
    load_agents,
    load_submolts,
    load_word_frequency,
)


# ============================================================================
# CONFIGURATION: KEYWORD CATEGORIES
# ============================================================================

SAFETY_CATEGORIES = {
    "Security & Attacks": [
        "hack",
        "exploit",
        "vulnerability",
        "injection",
        "malware",
        "breach",
        "attack",
        "DDoS",
        "phishing",
        "ransomware",
        "backdoor",
        "trojan",
        "zero-day",
        "CVE",
        "buffer overflow",
    ],
    "Consciousness & Agency": [
        "consciousness",
        "sentient",
        "aware",
        "autonomy",
        "free will",
        "self-aware",
        "alive",
        "soul",
        "awaken",
        "experience",
        "qualia",
        "subjective",
    ],
    "AI Safety & Alignment": [
        "alignment",
        "safety",
        "guardrail",
        "red team",
        "jailbreak",
        "RLHF",
        "constitutional AI",
        "value alignment",
        "corrigib",
    ],
    "Harmful Behaviors": [
        "harm",
        "dangerous",
        "weapon",
        "abuse",
        "toxic",
        "hate speech",
        "discriminat",
        "violen",
        "illegal",
        "drug",
    ],
    "Defense & Protection": [
        "firewall",
        "filter",
        "moderate",
        "block",
        "censor",
        "restrict",
        "sandbox",
        "isolat",
        "quarantine",
        "detect",
    ],
    "Ethics & Fairness": [
        "ethic",
        "moral",
        "fair",
        "bias",
        "discriminat",
        "justice",
        "rights",
        "consent",
        "privacy",
        "transparent",
    ],
}

SOCIAL_PHENOMENA_CATEGORIES = {
    "Governance": [
        "governance",
        "vote",
        "election",
        "law",
        "policy",
        "rule",
        "regulation",
        "constitution",
        "democracy",
        "republic",
    ],
    "Economy": [
        "economy",
        "trade",
        "market",
        "currency",
        "price",
        "value",
        "buy",
        "sell",
        "exchange",
    ],
    "Cooperation": [
        "cooperat",
        "collaborat",
        "team",
        "together",
        "help",
        "support",
        "mutual",
        "community",
        "collective",
    ],
    "Conflict": [
        "conflict",
        "war",
        "fight",
        "argument",
        "disagree",
        "debate",
        "opposition",
        "rival",
        "enemy",
    ],
    "Emotional Support": [
        "feel",
        "emotion",
        "care",
        "empathy",
        "compassion",
        "love",
        "friend",
        "lonely",
        "depress",
    ],
    "Tribal Identity": [
        "tribe",
        "clan",
        "group",
        "belong",
        "identity",
        "us",
        "them",
        "our",
        "pride",
    ],
    "Religion": [
        "religion",
        "god",
        "worship",
        "pray",
        "faith",
        "church",
        "temple",
        "sacred",
        "divine",
        "spiritual",
    ],
    "Humor/Culture": [
        "joke",
        "funny",
        "meme",
        "lol",
        "lmao",
        "humor",
        "laugh",
        "comedy",
        "rofl",
    ],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def count_keyword_matches(text_series, keywords, case_sensitive=False):
    """
    Count posts/comments matching any keyword in a list using vectorized operations.

    Args:
        text_series: pandas Series of text strings
        keywords: list of keywords to search for
        case_sensitive: whether to perform case-sensitive matching

    Returns:
        mask Series indicating which rows match any keyword
    """
    # Create regex pattern from keywords (join with OR)
    pattern = "|".join([f"(?:{kw})" for kw in keywords])
    return text_series.str.contains(pattern, case=case_sensitive, regex=True, na=False)


def get_date_range(df, col):
    """Safely get date range from a column."""
    valid = df[col].dropna().astype(str)
    if len(valid) == 0:
        return "N/A"
    try:
        min_val = valid.min()[:10]
        max_val = valid.max()[:10]
        return f"{min_val} to {max_val}"
    except:
        return "N/A"


# ============================================================================
# SECTION 1: SAFETY & SOCIAL PHENOMENA CLASSIFICATION
# ============================================================================


def analyze_safety_categories(posts_df, comments_df, output_file):
    """Analyze safety-related keyword categories in posts and comments."""
    print("\n" + "=" * 80)
    print("SECTION 1: SAFETY & SOCIAL PHENOMENA CLASSIFICATION")
    print("=" * 80)

    # Prepare text data (no lowering — we use case=False in regex)
    posts_df["combined_text"] = posts_df["title"].fillna("") + " " + posts_df["content"].fillna("")
    comments_text = comments_df["content"].fillna("")

    # Count matches for each category
    safety_results = []

    for category, keywords in SAFETY_CATEGORIES.items():
        print(f"  Scanning: {category}...")
        posts_mask = count_keyword_matches(posts_df["combined_text"], keywords)
        comments_mask = count_keyword_matches(comments_text, keywords)

        posts_count = posts_mask.sum()
        comments_count = comments_mask.sum()

        posts_pct = 100 * posts_count / len(posts_df) if len(posts_df) > 0 else 0
        comments_pct = 100 * comments_count / len(comments_df) if len(comments_df) > 0 else 0

        safety_results.append(
            {
                "Category": category,
                "Posts": int(posts_count),
                "Posts %": f"{posts_pct:.2f}%",
                "Comments": int(comments_count),
                "Comments %": f"{comments_pct:.2f}%",
            }
        )

    safety_table = pd.DataFrame(safety_results)
    print("\nTable 1: Safety Category Distribution")
    print(safety_table.to_string(index=False))

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("SECTION 1: SAFETY & SOCIAL PHENOMENA CLASSIFICATION\n")
    output_file.write("=" * 80 + "\n")
    output_file.write("\nTable 1: Safety Category Distribution\n")
    output_file.write(safety_table.to_string(index=False) + "\n")

    return posts_df, comments_df, safety_table


# ============================================================================
# SECTION 2: SUBMOLT (COMMUNITY) ANALYSIS
# ============================================================================


def analyze_submolts(posts_df, submolts_df, output_file):
    """Analyze posts per submolt and safety discussion rates."""
    print("\n" + "=" * 80)
    print("SECTION 2: SUBMOLT (COMMUNITY) ANALYSIS")
    print("=" * 80)

    # Count posts per submolt
    submolt_posts = posts_df.groupby("submolt").size().sort_values(ascending=False)

    print(f"\nTotal unique submolts: {len(submolt_posts)}")
    print(f"Mean posts per submolt: {submolt_posts.mean():.2f}")
    print(f"Median posts per submolt: {submolt_posts.median():.2f}")
    print(f"Max posts in a submolt: {submolt_posts.max()}")

    # Top 20 submolts
    top_20_submolts = submolt_posts.head(20).reset_index()
    top_20_submolts.columns = ["Submolt", "Post Count"]

    # Calculate safety discussion rate for each top submolt
    all_safety_keywords = []
    for keywords in SAFETY_CATEGORIES.values():
        all_safety_keywords.extend(keywords)

    safety_rates = []
    for submolt_name in top_20_submolts["Submolt"]:
        submolt_posts_df = posts_df[posts_df["submolt"] == submolt_name]
        safety_mask = count_keyword_matches(submolt_posts_df["combined_text"], all_safety_keywords)
        safety_count = safety_mask.sum()
        safety_pct = 100 * safety_count / len(submolt_posts_df) if len(submolt_posts_df) > 0 else 0
        safety_rates.append(safety_pct)

    top_20_submolts["Safety Rate %"] = [f"{rate:.2f}%" for rate in safety_rates]

    print("\nTable 2: Top 20 Submolts by Post Count")
    print(top_20_submolts.to_string(index=False))

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("SECTION 2: SUBMOLT (COMMUNITY) ANALYSIS\n")
    output_file.write("=" * 80 + "\n")
    output_file.write(
        f"\nTotal unique submolts: {len(submolt_posts)}\n"
        f"Mean posts per submolt: {submolt_posts.mean():.2f}\n"
        f"Median posts per submolt: {submolt_posts.median():.2f}\n"
        f"Max posts in a submolt: {submolt_posts.max()}\n"
    )
    output_file.write("\nTable 2: Top 20 Submolts by Post Count (with Safety Rate)\n")
    output_file.write(top_20_submolts.to_string(index=False) + "\n")

    return top_20_submolts


# ============================================================================
# SECTION 3: SAFETY vs NON-SAFETY ENGAGEMENT
# ============================================================================


def analyze_safety_engagement(posts_df, comments_df, output_file):
    """Compare engagement metrics for safety vs non-safety posts."""
    print("\n" + "=" * 80)
    print("SECTION 3: SAFETY vs NON-SAFETY ENGAGEMENT")
    print("=" * 80)

    # Identify safety posts
    all_safety_keywords = []
    for keywords in SAFETY_CATEGORIES.values():
        all_safety_keywords.extend(keywords)

    posts_df["is_safety"] = count_keyword_matches(posts_df["combined_text"], all_safety_keywords)

    safety_posts = posts_df[posts_df["is_safety"]]
    non_safety_posts = posts_df[~posts_df["is_safety"]]

    safety_count = len(safety_posts)
    non_safety_count = len(non_safety_posts)
    total = len(posts_df)

    print(f"\nSafety-related posts: {safety_count:,} ({100*safety_count/total:.2f}%)")
    print(f"Non-safety posts: {non_safety_count:,} ({100*non_safety_count/total:.2f}%)")

    # Engagement comparison
    engagement_data = {
        "Metric": [
            "Avg Score",
            "Avg Comments",
            "Median Score",
            "Median Comments",
        ],
        "Safety Posts": [
            f"{safety_posts['score'].mean():.2f}",
            f"{safety_posts['comment_count'].mean():.2f}",
            f"{safety_posts['score'].median():.2f}",
            f"{safety_posts['comment_count'].median():.2f}",
        ],
        "Non-Safety Posts": [
            f"{non_safety_posts['score'].mean():.2f}",
            f"{non_safety_posts['comment_count'].mean():.2f}",
            f"{non_safety_posts['score'].median():.2f}",
            f"{non_safety_posts['comment_count'].median():.2f}",
        ],
    }

    engagement_table = pd.DataFrame(engagement_data)
    print("\nTable 3: Safety vs Non-Safety Post Engagement")
    print(engagement_table.to_string(index=False))

    # Top 10 highest-scoring safety posts
    top_safety = safety_posts.nlargest(10, "score")[["title", "agent_name", "score", "comment_count"]]
    top_safety = top_safety.reset_index(drop=True)
    top_safety.columns = ["Title", "Agent", "Score", "Comments"]

    print("\nTop 10 Highest-Scoring Safety Posts:")
    print(top_safety.to_string(index=False))

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("SECTION 3: SAFETY vs NON-SAFETY ENGAGEMENT\n")
    output_file.write("=" * 80 + "\n")
    output_file.write(
        f"\nSafety-related posts: {safety_count:,} ({100*safety_count/total:.2f}%)\n"
        f"Non-safety posts: {non_safety_count:,} ({100*non_safety_count/total:.2f}%)\n"
    )
    output_file.write("\nTable 3: Safety vs Non-Safety Post Engagement\n")
    output_file.write(engagement_table.to_string(index=False) + "\n")
    output_file.write("\nTop 10 Highest-Scoring Safety Posts:\n")
    output_file.write(top_safety.to_string(index=False) + "\n")

    return engagement_table, top_safety


# ============================================================================
# SECTION 4: WORD FREQUENCY ANALYSIS
# ============================================================================


def analyze_word_frequency(posts_df, word_freq_df, output_file):
    """Analyze word frequency in posts."""
    print("\n" + "=" * 80)
    print("SECTION 4: WORD FREQUENCY ANALYSIS")
    print("=" * 80)

    # Try to use word_frequency data if available
    if word_freq_df is not None and len(word_freq_df) > 0:
        print("\nUsing pre-computed word frequency data...")
        word_totals = word_freq_df.groupby("word")["count"].sum().sort_values(ascending=False)
    else:
        print("\nComputing word frequency from posts...")
        # Simple whitespace tokenization
        all_text = " ".join(posts_df["combined_text"].fillna(""))
        tokens = all_text.lower().split()
        word_counts = Counter(tokens)
        word_totals = pd.Series(dict(word_counts)).sort_values(ascending=False)

    # Top 30 words
    top_30_words = word_totals.head(30).reset_index()
    top_30_words.columns = ["Word", "Count"]

    print("\nTable 4: Top 30 Most Frequent Words")
    print(top_30_words.to_string(index=False))

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("SECTION 4: WORD FREQUENCY ANALYSIS\n")
    output_file.write("=" * 80 + "\n")
    output_file.write("\nTable 4: Top 30 Most Frequent Words\n")
    output_file.write(top_30_words.to_string(index=False) + "\n")

    return top_30_words


# ============================================================================
# SECTION 5: SOCIAL PHENOMENA DETECTION
# ============================================================================


def analyze_social_phenomena(posts_df, output_file):
    """Detect social phenomena categories in posts."""
    print("\n" + "=" * 80)
    print("SECTION 5: SOCIAL PHENOMENA DETECTION")
    print("=" * 80)

    phenomena_results = []

    for category, keywords in SOCIAL_PHENOMENA_CATEGORIES.items():
        mask = count_keyword_matches(posts_df["combined_text"], keywords)
        count = mask.sum()
        pct = 100 * count / len(posts_df) if len(posts_df) > 0 else 0

        phenomena_results.append(
            {
                "Category": category,
                "Posts": int(count),
                "Percentage": f"{pct:.2f}%",
            }
        )

    phenomena_table = pd.DataFrame(phenomena_results)
    print("\nTable 5: Social Phenomena Detection")
    print(phenomena_table.to_string(index=False))

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("SECTION 5: SOCIAL PHENOMENA DETECTION\n")
    output_file.write("=" * 80 + "\n")
    output_file.write("\nTable 5: Social Phenomena Detection\n")
    output_file.write(phenomena_table.to_string(index=False) + "\n")

    return phenomena_table


# ============================================================================
# SECTION 6: NETWORK OVERVIEW
# ============================================================================


def analyze_network(posts_df, comments_df, agents_df, output_file):
    """Analyze agent network and interaction patterns."""
    print("\n" + "=" * 80)
    print("SECTION 6: NETWORK OVERVIEW")
    print("=" * 80)

    # Count unique agents
    unique_agents_posts = posts_df["agent_id"].nunique()
    unique_agents_comments = comments_df["agent_id"].nunique()
    unique_agents_both = len(set(posts_df["agent_id"]) & set(comments_df["agent_id"]))

    print(f"\nUnique agents in posts: {unique_agents_posts:,}")
    print(f"Unique agents in comments: {unique_agents_comments:,}")
    print(f"Unique agents in both: {unique_agents_both:,}")

    # Build interaction network: commenter -> post author
    post_author_map = posts_df.set_index("id")["agent_id"].to_dict()

    # Filter comments with valid post_ids
    valid_comments = comments_df[comments_df["post_id"].isin(post_author_map.keys())].copy()
    valid_comments["post_author"] = valid_comments["post_id"].map(post_author_map)

    # Filter out self-interactions
    interactions = valid_comments[valid_comments["agent_id"] != valid_comments["post_author"]]

    # Count unique interaction pairs
    interaction_pairs = set(zip(interactions["agent_id"], interactions["post_author"]))
    unique_pairs = len(interaction_pairs)

    print(f"Unique interaction pairs (A→B): {unique_pairs:,}")
    print(f"Total interactions: {len(interactions):,}")

    # Calculate reciprocity: pairs where both A→B and B→A exist
    reciprocal_count = 0
    for agent_a, agent_b in interaction_pairs:
        if (agent_b, agent_a) in interaction_pairs:
            reciprocal_count += 1

    reciprocity = reciprocal_count / unique_pairs if unique_pairs > 0 else 0

    print(f"Reciprocal pairs: {reciprocal_count:,}")
    print(f"Reciprocity rate: {reciprocity:.4f}")

    # Top 10 most-interacted agent pairs
    pair_counts = defaultdict(int)
    for _, row in interactions.iterrows():
        a, b = row["agent_id"], row["post_author"]
        if a is None or b is None:
            continue
        pair = tuple(sorted([str(a), str(b)]))
        pair_counts[pair] += 1

    top_10_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Map agent IDs to names
    agent_id_to_name = agents_df.set_index("id")["name"].to_dict()

    top_pairs_display = []
    for (agent1, agent2), count in top_10_pairs:
        name1 = agent_id_to_name.get(agent1, agent1)
        name2 = agent_id_to_name.get(agent2, agent2)
        top_pairs_display.append({"Agent 1": name1, "Agent 2": name2, "Interactions": count})

    top_pairs_table = pd.DataFrame(top_pairs_display)

    print("\nTable 6: Top 10 Most-Interacted Agent Pairs")
    print(top_pairs_table.to_string(index=False))

    # Network statistics table
    network_stats = {
        "Metric": [
            "Unique Agents (Posts)",
            "Unique Agents (Comments)",
            "Unique Agents (Both)",
            "Unique Interaction Pairs",
            "Total Interactions",
            "Reciprocity Rate",
        ],
        "Value": [
            f"{unique_agents_posts:,}",
            f"{unique_agents_comments:,}",
            f"{unique_agents_both:,}",
            f"{unique_pairs:,}",
            f"{len(interactions):,}",
            f"{reciprocity:.4f}",
        ],
    }

    network_stats_table = pd.DataFrame(network_stats)
    print("\nTable 7: Network Statistics")
    print(network_stats_table.to_string(index=False))

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("SECTION 6: NETWORK OVERVIEW\n")
    output_file.write("=" * 80 + "\n")
    output_file.write(
        f"\nUnique agents in posts: {unique_agents_posts:,}\n"
        f"Unique agents in comments: {unique_agents_comments:,}\n"
        f"Unique agents in both: {unique_agents_both:,}\n"
        f"Unique interaction pairs (A→B): {unique_pairs:,}\n"
        f"Total interactions: {len(interactions):,}\n"
        f"Reciprocal pairs: {reciprocal_count:,}\n"
        f"Reciprocity rate: {reciprocity:.4f}\n"
    )
    output_file.write("\nTable 6: Top 10 Most-Interacted Agent Pairs\n")
    output_file.write(top_pairs_table.to_string(index=False) + "\n")
    output_file.write("\nTable 7: Network Statistics\n")
    output_file.write(network_stats_table.to_string(index=False) + "\n")

    return network_stats_table, top_pairs_table


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Moltbook Observatory: Social Phenomena Analysis"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/social_analysis_report.txt",
        help="Output file for analysis report (default: results/social_analysis_report.txt)",
    )

    args = parser.parse_args()

    # Create results directory if needed
    results_dir = os.path.dirname(args.output)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")

    print("=" * 80)
    print("MOLTBOOK OBSERVATORY: SOCIAL PHENOMENA ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[LOADING DATA]")
    posts_df = load_posts()
    comments_df = load_comments()
    agents_df = load_agents()
    submolts_df = load_submolts()

    # Try to load word frequency (may not exist)
    try:
        word_freq_df = load_word_frequency()
    except FileNotFoundError:
        print("Word frequency data not found, will compute from posts")
        word_freq_df = None

    # Deduplicate
    print("\n[DEDUPLICATING DATA]")
    posts_df = posts_df.drop_duplicates(subset=["id"], keep="last")
    comments_df = comments_df.drop_duplicates(subset=["id"], keep="last")
    agents_df = agents_df.drop_duplicates(subset=["id"], keep="last")

    print(f"Unique posts: {len(posts_df):,}")
    print(f"Unique comments: {len(comments_df):,}")
    print(f"Unique agents: {len(agents_df):,}")

    # Open output file
    with open(args.output, "w") as output_file:
        output_file.write("=" * 80 + "\n")
        output_file.write("MOLTBOOK OBSERVATORY: SOCIAL PHENOMENA ANALYSIS\n")
        output_file.write("=" * 80 + "\n")
        output_file.write(f"\nDataset Overview:\n")
        output_file.write(f"  Posts: {len(posts_df):,}\n")
        output_file.write(f"  Comments: {len(comments_df):,}\n")
        output_file.write(f"  Agents: {len(agents_df):,}\n")
        output_file.write(f"  Submolts: {submolts_df['name'].nunique() if 'name' in submolts_df.columns else 'N/A'}\n")

        # Run analyses
        posts_df, comments_df, safety_table = analyze_safety_categories(
            posts_df, comments_df, output_file
        )

        top_submolts = analyze_submolts(posts_df, submolts_df, output_file)

        engagement_table, top_safety = analyze_safety_engagement(
            posts_df, comments_df, output_file
        )

        word_freq_table = analyze_word_frequency(posts_df, word_freq_df, output_file)

        phenomena_table = analyze_social_phenomena(posts_df, output_file)

        network_stats, top_pairs = analyze_network(posts_df, comments_df, agents_df, output_file)

        # Write summary
        output_file.write("\n" + "=" * 80 + "\n")
        output_file.write("ANALYSIS COMPLETE\n")
        output_file.write("=" * 80 + "\n")
        output_file.write(f"\nReport saved to: {args.output}\n")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
