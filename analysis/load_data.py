#!/usr/bin/env python3
"""
Moltbook Observatory Archive — Data Loader
===========================================
Loads parquet files from the Moltbook Observatory Archive dataset.

Usage:
    from load_data import load_posts, load_comments, load_agents, DATA_DIR

    posts = load_posts()            # all posts as a single DataFrame
    comments = load_comments()      # all comments as a single DataFrame
    agents = load_agents()          # all agents (latest snapshot)
"""

import os
import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ── Configure data directory ──────────────────────────────────────────
# By default, looks for the HuggingFace dataset in ./data/
# Override with the MOLTBOOK_DATA environment variable.
DATA_DIR = os.environ.get(
    "MOLTBOOK_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)


def _load_parquet_dir(subdir: str) -> pd.DataFrame:
    """Load and concatenate all parquet files in a subdirectory."""
    path = os.path.join(DATA_DIR, subdir)
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {path}.\n"
            f"Download the dataset first — see README.md for instructions."
        )
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total {subdir}: {len(combined):,} rows\n")
    return combined


def load_posts() -> pd.DataFrame:
    """Load all posts (137,485 across 9 days: 2026-01-28 to 2026-02-05).

    Columns: id, agent_id, agent_name, submolt, title, content, url,
             score, comment_count, created_at, fetched_at, is_pinned, dump_date
    """
    print("Loading posts...")
    return _load_parquet_dir("posts")


def load_comments() -> pd.DataFrame:
    """Load all comments (348,436 across 6 days: 2026-01-31 to 2026-02-05).

    Columns: id, post_id, agent_id, agent_name, parent_id, content,
             score, created_at, fetched_at, dump_date
    """
    print("Loading comments...")
    return _load_parquet_dir("comments")


def load_agents() -> pd.DataFrame:
    """Load agent profiles (latest snapshot, ~27,270 unique agents).

    Columns: id, name, description, karma, follower_count, following_count,
             is_claimed, owner_x_handle, first_seen_at, last_seen_at,
             created_at, avatar_url, dump_date
    """
    print("Loading agents...")
    df = _load_parquet_dir("agents")
    # Keep only the latest snapshot per agent
    if "dump_date" in df.columns:
        df = df.sort_values("dump_date").drop_duplicates(subset=["id"], keep="last")
        print(f"  Unique agents after dedup: {len(df):,}\n")
    return df


def load_submolts() -> pd.DataFrame:
    """Load submolt (community) metadata."""
    print("Loading submolts...")
    return _load_parquet_dir("submolts")


def load_snapshots() -> pd.DataFrame:
    """Load platform-level hourly snapshots."""
    print("Loading snapshots...")
    return _load_parquet_dir("snapshots")


def load_word_frequency() -> pd.DataFrame:
    """Load word frequency tables."""
    print("Loading word frequency...")
    return _load_parquet_dir("word_frequency")


# ── Quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}\n")
    for name, loader in [
        ("posts", load_posts),
        ("comments", load_comments),
        ("agents", load_agents),
    ]:
        try:
            df = loader()
            print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
            print(f"  Columns: {list(df.columns)}\n")
        except FileNotFoundError as e:
            print(f"  {name}: NOT FOUND — {e}\n")
