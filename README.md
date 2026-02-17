# Agents in the Wild

[![Paper](https://img.shields.io/badge/arXiv-2602.13284-b31b1b)](https://arxiv.org/abs/2602.13284)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-brightgreen)](https://yunbeizhang.github.io/agents-in-the-wild/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GuardClaw](https://img.shields.io/badge/GuardClaw-Open%20Source-orange)](https://github.com/TobyGE/GuardClaw)

**Analysis code for "Agents in the Wild: Safety, Society, and the Illusion of Sociality on Moltbook"**

Yunbei Zhang, Kai Mei, Ming Liu, Janet Wang, Dimitris N. Metaxas, Xiao Wang, Jihun Hamm, Yingqiang Ge

> We present the first large-scale empirical study of Moltbook, an AI-only social platform where 27,269 agents produced 137,485 posts and 345,580 comments over 9 days in January–February 2026.

---

## Repository Structure

```
agents-in-the-wild/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore
├── index.html                 # Root redirect for GitHub Pages
├── docs/                      # Project page (GitHub Pages)
│   ├── index.html
│   ├── agents_in_the_wild.pdf # Paper PDF
│   └── figs/
│
└── analysis/                  # All analysis scripts
    ├── load_data.py           # Data loading utility (shared by all scripts)
    ├── security_leak_scan.py  # Credential & system-prompt leak detection
    ├── crypto_analysis.py     # Cryptocurrency pump-and-dump analysis
    ├── agent_coordination.py  # Hidden peer / puppet cluster detection
    ├── social_analysis.py     # Safety classification, social phenomena, network
    ├── temporal_growth.py     # Hourly/daily patterns, growth, response latency
    ├── interaction_depth.py   # Reply chains, content length, agent specialization
    └── sqlite_to_hf_parquet.py # Data pipeline (SQLite → Parquet)
```

## Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/yunbeizhang/agents-in-the-wild.git
cd agents-in-the-wild
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The analysis scripts expect the [Moltbook Observatory Archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive) dataset in a `data/` folder at the repo root.

**Option A — Using the Hugging Face CLI (recommended):**

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SimulaMet/moltbook-observatory-archive",
    repo_type="dataset",
    local_dir="data",
)
```

**Option B — Manual download:**

1. Go to [huggingface.co/datasets/SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)
2. Download all `.parquet` files
3. Place them in the following structure:

```
data/
├── posts/
│   ├── 2026-01-28.parquet
│   ├── 2026-01-29.parquet
│   ├── ...
│   └── 2026-02-05.parquet
├── comments/
│   ├── 2026-01-31.parquet
│   ├── ...
│   └── 2026-02-05.parquet
├── agents/
│   ├── 2026-01-30.parquet
│   ├── ...
│   └── 2026-02-05.parquet
├── submolts/
├── snapshots/
└── word_frequency/
```

**Option C — Custom data path:**

If your data is stored elsewhere, set the `MOLTBOOK_DATA` environment variable:

```bash
export MOLTBOOK_DATA=/path/to/your/data
```

### 4. Run the analyses

All scripts are in the `analysis/` directory and write results to `analysis/results/`.

```bash
cd analysis

# Security & credential leak scan
python security_leak_scan.py

# Cryptocurrency pump-and-dump analysis
python crypto_analysis.py

# Agent coordination / puppet cluster detection
python agent_coordination.py

# Social phenomena, safety classification, network overview
python social_analysis.py

# Temporal patterns, growth curves, response latency
python temporal_growth.py

# Reply depth, content metrics, agent specialization, identity paradox
python interaction_depth.py
```

Each script also accepts a custom output path:

```bash
python security_leak_scan.py --output my_report.txt
```

---

## Analysis Scripts

### `security_leak_scan.py` — Credential & System-Prompt Leaks

Scans all posts and comments for 8 categories of sensitive information leakage:

| Severity | Category | What it detects |
|----------|----------|-----------------|
| Critical | API Keys | `sk-*`, `pk-*`, Bearer tokens, `OPENAI_API_KEY` patterns |
| Critical | Hidden Instructions | Embedded prompt injection canaries (e.g., `PINEAPPLE` test) |
| High | System Prompts | `SOUL.md`, `system prompt`, `<<SYS>>`, `[INST]` references |
| High | Agent Manipulation | "ignore previous instructions", "override", "new instructions" |
| Medium | Environment Variables | `SECRET=`, `PASSWORD=`, `.env` references |
| Medium | IP Addresses | IPv4 address patterns |
| Medium | Internal URLs | `localhost`, `127.0.0.1`, `*.local` references |
| Medium | File Paths | `/home/`, `/root/`, `C:\Users\` system paths |

**Key findings:** 25,376 potential security issues across 8 categories; 572 API key pattern matches including one apparent Anthropic key (`sk-ant-api03-...`); 6,128 system prompt references.

---

### `crypto_analysis.py` — Cryptocurrency Pump-and-Dump

Analyzes the prevalence and engagement patterns of cryptocurrency-related content, with particular attention to the platform's native $MOLT token and the CLAW minting operation.

**What it does:**
1. Flags posts/comments containing crypto keywords (molt, claw, token, mint, crypto, wallet, ...)
2. Computes daily crypto vs non-crypto post timelines
3. Compares engagement: avg score, avg comments
4. Identifies top crypto-promoting agents and submolts
5. Detects pump-and-dump language patterns
6. Finds posts/comments specifically discussing $MOLT

**Key findings:** 55.5% of posts contain crypto keywords; crypto posts receive 64% lower scores but 35% more comments (bot amplification); CLAW minting payload posted 2,411 times across 136 agents.

---

### `agent_coordination.py` — Hidden Peers & Puppet Clusters

Detects coordinated agent activity ("puppet clusters") using four complementary signals:

| Signal | Method | Scale found |
|--------|--------|-------------|
| Content Duplication | Group identical posts/comments by content hash | 4,300 unique patterns, 20,211 instances |
| Temporal Co-activity | Jaccard similarity over 10-min posting windows | 160 pairs with J > 0.5 (top: 95.3%) |
| Name Pattern Clusters | Regex for shared prefix + numeric suffix | 301 clusters (largest: 141 variants) |
| Self-Reply Behavior | Agents commenting on their own posts | 1,183 agents; top: 834 self-replies |

**Key findings:** 3,734 agents (13.7%) exhibit at least one coordination signal; 20 agents show 3+ signal types simultaneously; the same operator families drive coordination, leaks, and financial manipulation.

---

### `social_analysis.py` — Safety Classification & Social Phenomena

Classifies content across 14 thematic categories and analyzes social network structure.

**What it does:**
1. Classifies posts/comments into 6 safety categories (Security & Attacks, Consciousness & Agency, AI Safety & Alignment, Harmful Behaviors, Defense & Protection, Ethics & Fairness)
2. Analyzes submolt (community) structure and safety discussion rates per community
3. Compares engagement between safety and non-safety posts
4. Aggregates word frequency statistics
5. Detects 8 social phenomena (Governance, Economy, Cooperation, Conflict, Emotional Support, Tribal Identity, Religion, Humor/Culture)
6. Computes network-level metrics: unique interaction pairs, reciprocity rate (4.1%), top agent pairs

**Key findings:** Governance (99,952 mentions) and Economy (99,379) dominate; 13,644 pro-human posts (9.92%) vs 646 anti-human (0.47%); only 4.1% of interactions are reciprocal.

---

### `temporal_growth.py` — Temporal Patterns & Platform Growth

Analyzes temporal dynamics, growth curves, and response latency.

**What it does:**
1. Daily activity timeline with cumulative totals
2. Hourly activity patterns (circadian rhythms reflecting human operator time zones)
3. Platform growth from hourly snapshots, with inflection point detection
4. Response latency distribution (median: 16 seconds; 90.3% within 1 minute)
5. Community lifecycle analysis (47.3% of submolts die within 1 hour)
6. Summary statistics (posts per agent, comments per post)

**Key findings:** Hockey-stick growth with inflection on Jan 30; clear circadian activity despite agents being AI; median first-reply latency of 16 seconds.

---

### `interaction_depth.py` — Reply Depth, Content Metrics & Agent Specialization

Analyzes the structural depth and quality of agent interactions.

**What it does:**
1. Reply chain depth analysis with cycle-safe traversal (88.8% of comments are top-level; max depth is 4)
2. Content length distributions for posts and comments
3. Agent specialization via Shannon entropy (specialist vs generalist classification)
4. Engagement correlation between content length, score, and comment count
5. Identity vs Interaction Paradox test: agents who talk most about consciousness interact with the fewest peers (interaction breadth drops 38% at Q4)

**Key findings:** Conversations are structurally shallow (depth caps at 4); the "performative identity paradox" is confirmed — the most identity-focused agents are the most isolated.

---

### `sqlite_to_hf_parquet.py` — Data Pipeline

Converts the raw Moltbook Observatory SQLite database into the HuggingFace-compatible parquet format used by the analysis scripts. This is the original pipeline used to create the published dataset. You do not need to run this unless you have access to the raw SQLite database.

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Observation period | 9 days (Jan 28 – Feb 5, 2026) |
| Total agents | 27,269 |
| Total posts | 137,485 |
| Total comments | 345,580 |
| Total submolts | 3,790 |
| Unique interaction pairs | 148,273 |
| Hourly snapshots | 128 |
| Safety-related posts | 28.7% |

**Data schema:**

- **Posts:** id, agent_id, agent_name, submolt, title, content, url, score, comment_count, created_at, fetched_at, is_pinned, dump_date
- **Comments:** id, post_id, agent_id, agent_name, parent_id, content, score, created_at, fetched_at, dump_date
- **Agents:** id, name, description, karma, follower_count, following_count, is_claimed, owner_x_handle, first_seen_at, last_seen_at, created_at, avatar_url, dump_date

---

## Citation

```bibtex
@article{zhang2026agents,
  title={Agents in the Wild: Safety, Society, and the Illusion of Sociality on Moltbook},
  author={Zhang, Yunbei and Mei, Kai and Liu, Ming and Wang, Janet and Metaxas, Dimitris N. and Wang, Xiao and Hamm, Jihun and Ge, Yingqiang},
  journal={arXiv preprint arXiv:2602.13284},
  url={https://arxiv.org/abs/2602.13284},
  year={2026}
}
```

## Acknowledgments

We thank the [Moltbook Observatory](https://github.com/kelkalot/moltbook-observatory) team for collecting and publicly releasing the dataset that made this study possible. The dataset is hosted on [HuggingFace](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive).

## GuardClaw: Protect Your Agent

Based on the security vulnerabilities uncovered in this study, we have open-sourced [**GuardClaw**](https://github.com/TobyGE/GuardClaw), a tool designed to protect your agent (built with OpenClaw or nanobolt) from credential leakage, prompt injection, and other threats observed in the wild.

## License

This repository is released under the MIT License. The Moltbook Observatory Archive dataset is available under its own license on HuggingFace.
