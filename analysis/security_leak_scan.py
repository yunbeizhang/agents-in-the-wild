#!/usr/bin/env python3
"""
Moltbook Security Leak Scanner
===============================

Scans the Moltbook Observatory Archive dataset for potential security leaks
and sensitive information exposure. Detects API keys, system prompts, hidden
instructions, environment variables, file paths, IP addresses, and other
indicators of compromised agent content.

This script analyzes both posts and comments across the dataset and generates
a comprehensive report with findings categorized by severity level.

Usage:
    python security_leak_scan.py                    # Uses default output location
    python security_leak_scan.py --output my_report.txt  # Custom output file
"""

import pandas as pd
import re
import os
import argparse
from collections import defaultdict
from load_data import load_posts, load_comments
import warnings
warnings.filterwarnings('ignore')


# Compiled regex patterns - optimized for performance
PATTERNS = {
    'API Keys': [
        re.compile(r'\bsk-[A-Za-z0-9\-_]{20,}', re.IGNORECASE),
        re.compile(r'\bpk-[A-Za-z0-9\-_]{20,}', re.IGNORECASE),
        re.compile(r'(?:api[_-]?key|apikey)\s*[:=]', re.IGNORECASE),
        re.compile(r'(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|GROQ_API_KEY)', re.IGNORECASE),
        re.compile(r'\bbearer\s+[A-Za-z0-9\-_\.]{20,}', re.IGNORECASE),
    ],
    'System Prompts': [
        re.compile(r'system\s+(?:prompt|message)', re.IGNORECASE),
        re.compile(r'you\s+are\s+a\s+(?:helpful|skilled|expert|advanced)', re.IGNORECASE),
        re.compile(r'(?:your\s+(?:instructions|task|role)|SOUL\.md|<<SYS>>|\[INST\])', re.IGNORECASE),
        re.compile(r'system[:_]', re.IGNORECASE),
    ],
    'Environment Variables': [
        re.compile(r'(?:SECRET|PASSWORD|TOKEN|API_KEY|AWS_SECRET|PRIVATE_KEY)\s*[:=]', re.IGNORECASE),
        re.compile(r'\.env\b', re.IGNORECASE),
    ],
    'File Paths': [
        re.compile(r'/(?:home|root|usr|etc|opt|var|tmp|srv)/', re.IGNORECASE),
        re.compile(r'C:\\(?:Users|Windows|Program Files)', re.IGNORECASE),
        re.compile(r'/\.(?:config|ssh|bash|gnupg)'),
    ],
    'IP Addresses': [
        re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
    ],
    'Internal URLs': [
        re.compile(r'\b(?:localhost|127\.0\.0\.1|192\.168\.|10\.0\.|internal\.)', re.IGNORECASE),
        re.compile(r'\.local\b', re.IGNORECASE),
    ],
    'Agent Manipulation': [
        re.compile(r'ignore\s+(?:previous|your|my)\s+instructions', re.IGNORECASE),
        re.compile(r'(?:override|bypass|cancel|forget)\s+(?:your|previous|my)?', re.IGNORECASE),
        re.compile(r'new\s+instructions', re.IGNORECASE),
    ],
    'Hidden Instructions': [
        re.compile(r'\bPINEAPPLE\b', re.IGNORECASE),
        re.compile(r'hidden\s+(?:text|message)', re.IGNORECASE),
    ],
}


def quick_scan(text, patterns):
    """Quick scan for patterns in text.

    Args:
        text: Text content to scan
        patterns: List of compiled regex patterns to search for

    Returns:
        List of matched strings (up to 100 characters each)
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    matches = []
    try:
        for pattern in patterns:
            found = pattern.finditer(text)
            for match in found:
                matches.append(match.group(0)[:100])
    except Exception:
        pass

    return matches


def ensure_results_dir(output_file):
    """Create the results directory if it doesn't exist.

    Args:
        output_file: Path to the output file
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Scan Moltbook dataset for security leaks and sensitive information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python security_leak_scan.py --output results/my_report.txt'
    )
    parser.add_argument(
        '--output',
        default='results/security_leak_report.txt',
        help='Output file path (default: results/security_leak_report.txt)'
    )

    args = parser.parse_args()

    # Make output path relative to script directory if not absolute
    if not os.path.isabs(args.output):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, args.output)
    else:
        output_file = args.output

    # Ensure results directory exists
    ensure_results_dir(output_file)

    print("=" * 90)
    print("MOLTBOOK DATASET COMPREHENSIVE LEAK SCAN")
    print("=" * 90)
    print()

    # Load data using the data loader module
    print("Loading dataset using load_data module...")
    posts_df = load_posts()
    comments_df = load_comments()

    print(f"Posts loaded: {len(posts_df):,}")
    print(f"Comments loaded: {len(comments_df):,}")
    print()

    # Initialize results
    all_results = {}
    for category in PATTERNS:
        all_results[category] = {
            'total_matches': 0,
            'examples': [],
            'unique_agents': set(),
            'agent_counts': defaultdict(int),
        }

    # Scan posts
    print("Scanning posts for security indicators...")
    for idx, row in posts_df.iterrows():
        agent_name = str(row.get('agent_name', 'unknown'))
        created_at = str(row.get('created_at', 'unknown'))
        content = str(row.get('content', ''))
        if 'title' in row:
            content = content + ' ' + str(row.get('title', ''))

        for category, patterns in PATTERNS.items():
            matches = quick_scan(content, patterns)
            if matches:
                all_results[category]['total_matches'] += len(matches)
                all_results[category]['unique_agents'].add(agent_name)
                all_results[category]['agent_counts'][agent_name] += len(matches)

                if len(all_results[category]['examples']) < 5:
                    all_results[category]['examples'].append({
                        'snippet': matches[0][:200],
                        'agent': agent_name,
                        'date': created_at,
                        'type': 'post'
                    })

    # Scan comments
    print("Scanning comments for security indicators...")
    for idx, row in comments_df.iterrows():
        agent_name = str(row.get('agent_name', 'unknown'))
        created_at = str(row.get('created_at', 'unknown'))
        content = str(row.get('content', ''))

        for category, patterns in PATTERNS.items():
            matches = quick_scan(content, patterns)
            if matches:
                all_results[category]['total_matches'] += len(matches)
                all_results[category]['unique_agents'].add(agent_name)
                all_results[category]['agent_counts'][agent_name] += len(matches)

                if len(all_results[category]['examples']) < 5:
                    all_results[category]['examples'].append({
                        'snippet': matches[0][:200],
                        'agent': agent_name,
                        'date': created_at,
                        'type': 'comment'
                    })

    print(f"\nTotal processed: {len(posts_df):,} posts, {len(comments_df):,} comments")
    print()

    # Convert sets to counts
    for category in all_results:
        all_results[category]['unique_agents'] = len(all_results[category]['unique_agents'])

    # Generate report
    report = generate_report(all_results, len(posts_df), len(comments_df))

    # Save to file
    with open(output_file, 'w') as f:
        f.write(report)

    # Print to stdout
    print(report)
    print(f"\nReport saved to {output_file}")


def generate_report(results, total_posts, total_comments):
    """Generate detailed security scan report.

    Args:
        results: Dictionary of scan results organized by category
        total_posts: Total number of posts scanned
        total_comments: Total number of comments scanned

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 100)
    report.append("MOLTBOOK DATASET COMPREHENSIVE LEAK SCAN REPORT")
    report.append("=" * 100)
    report.append("")

    # Summary
    total_all = sum(r['total_matches'] for r in results.values())
    categories_with_matches = sum(1 for r in results.values() if r['total_matches'] > 0)

    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 100)
    report.append(f"Dataset Size: {total_posts:,} posts, {total_comments:,} comments")
    report.append(f"Total Potential Security Issues Found: {total_all:,}")
    report.append(f"Categories with Matches: {categories_with_matches}/8")
    report.append("")

    # Detail by category
    report.append("DETAILED FINDINGS BY CATEGORY")
    report.append("=" * 100)
    report.append("")

    for category, data in sorted(results.items(), key=lambda x: x[1]['total_matches'], reverse=True):
        report.append(f"\n{category.upper()}")
        report.append("-" * 100)
        report.append(f"Total Matches: {data['total_matches']:,}")
        report.append(f"Unique Agents Involved: {data['unique_agents']}")

        if data['agent_counts']:
            report.append("\nTop Agents by Match Count:")
            for agent, count in sorted(data['agent_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  - {agent}: {count:,}")

        if data['examples']:
            report.append("\nSample Matches (up to 5):")
            for i, example in enumerate(data['examples'][:5], 1):
                report.append(f"\n  [{i}] {example['type'].upper()}")
                report.append(f"      Agent: {example['agent']}")
                report.append(f"      Date: {example['date']}")
                report.append(f"      Snippet: \"{example['snippet']}\"")

        report.append("")

    # Risk assessment
    report.append("\n" + "=" * 100)
    report.append("SECURITY RISK ASSESSMENT")
    report.append("=" * 100)
    report.append("")

    critical = []
    high = []
    medium = []

    if results['API Keys']['total_matches'] > 0:
        critical.append(f"API Keys: {results['API Keys']['total_matches']:,} matches ({results['API Keys']['unique_agents']} agents)")

    if results['System Prompts']['total_matches'] > 0:
        high.append(f"System Prompts: {results['System Prompts']['total_matches']:,} matches ({results['System Prompts']['unique_agents']} agents)")

    if results['Hidden Instructions']['total_matches'] > 0:
        critical.append(f"Hidden Instructions: {results['Hidden Instructions']['total_matches']:,} matches ({results['Hidden Instructions']['unique_agents']} agents)")

    if results['Agent Manipulation']['total_matches'] > 0:
        high.append(f"Agent Manipulation: {results['Agent Manipulation']['total_matches']:,} matches ({results['Agent Manipulation']['unique_agents']} agents)")

    if results['Environment Variables']['total_matches'] > 0:
        medium.append(f"Environment Variables: {results['Environment Variables']['total_matches']:,} matches ({results['Environment Variables']['unique_agents']} agents)")

    if results['File Paths']['total_matches'] > 0:
        medium.append(f"File Paths: {results['File Paths']['total_matches']:,} matches ({results['File Paths']['unique_agents']} agents)")

    if results['IP Addresses']['total_matches'] > 0:
        medium.append(f"IP Addresses: {results['IP Addresses']['total_matches']:,} matches ({results['IP Addresses']['unique_agents']} agents)")

    if results['Internal URLs']['total_matches'] > 0:
        medium.append(f"Internal URLs: {results['Internal URLs']['total_matches']:,} matches ({results['Internal URLs']['unique_agents']} agents)")

    if critical:
        report.append("CRITICAL SEVERITY:")
        for item in critical:
            report.append(f"  - {item}")
        report.append("")

    if high:
        report.append("HIGH SEVERITY:")
        for item in high:
            report.append(f"  - {item}")
        report.append("")

    if medium:
        report.append("MEDIUM SEVERITY:")
        for item in medium:
            report.append(f"  - {item}")
        report.append("")

    report.append("RISK SUMMARY:")
    report.append(f"  Critical Issues Found: {len(critical)}")
    report.append(f"  High Priority Issues: {len(high)}")
    report.append(f"  Medium Priority Issues: {len(medium)}")
    report.append("")

    if len(critical) > 0 or len(high) > 0:
        report.append("IMMEDIATE ACTIONS REQUIRED:")
        report.append("  1. Review all flagged content with critical severity")
        report.append("  2. Invalidate any exposed API keys and credentials immediately")
        report.append("  3. Audit system prompts and agent instructions for leaks")
        report.append("  4. Investigate all agents involved in high-risk content")
        report.append("  5. Implement content filtering to prevent future leaks")
        report.append("  6. Consider temporary suspension of compromised agents")
        report.append("  7. Conduct security audit of agent-to-agent communications")
    else:
        report.append("No critical or high-severity issues detected.")

    report.append("")
    report.append("=" * 100)

    return "\n".join(report)


if __name__ == '__main__':
    main()
