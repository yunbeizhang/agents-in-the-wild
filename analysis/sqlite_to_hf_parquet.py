from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sqlite3

try:
    from huggingface_hub import HfApi, create_repo
except Exception:
    HfApi = None
    create_repo = None


# ----------------------------
# Config
# ----------------------------

@dataclass
class ExportConfig:
    db_path: Path
    out_dir: Path = Path("hf_export")
    state_path: Optional[Path] = None

    # Hugging Face (optional)
    hf_repo_id: Optional[str] = None  # e.g. "username/observatory"
    hf_repo_type: str = "dataset"

    # Incremental selection preferences (first match wins if column exists)
    incremental_col_priority: Tuple[str, ...] = (
        "fetched_at",
        "last_seen_at",
        "updated_at",
        "timestamp",
        "hour",
        "first_seen_at",
        "created_at",
    )

    # Which column to use to derive the archival 'dump_date' (creation date)
    creation_col_priority: Tuple[str, ...] = (
        "first_seen_at",
        "created_at",
        "timestamp",
        "hour",
    )

    # Generic rolling backfill in days (helps capture updates without updated_at)
    default_backfill_days: int = 0

    # Per-table backfill overrides (generic for your schema)
    per_table_backfill_days: Dict[str, int] = None  # set in __post_init__

    # Verbose logging
    verbose: bool = False

    # Chunk size: keeps memory stable for big tables
    chunk_rows: int = 250_000

    def __post_init__(self):
        if self.state_path is None:
            self.state_path = self.out_dir / "state.json"
        if self.per_table_backfill_days is None:
            # Sensible defaults for your schema:
            # - posts/comments often get rescored/comment_count changes on re-fetch: small backfill
            # - submolts counts change; without a fetched_at, backfill helps
            # - others are append-only (snapshots, follows, word_frequency) so 0 is fine
            self.per_table_backfill_days = {
                "posts": 7,
                "comments": 7,
                "submolts": 30,
                "agents": 7,
            }


# ----------------------------
# SQLite introspection
# ----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def today_utc_date() -> str:
    return utc_now().date().isoformat()

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"last_exported": {}}

def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))

def list_tables(con: sqlite3.Connection) -> List[str]:
    # excludes sqlite internal tables
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
      AND name NOT LIKE 'sqlite_%'
    ORDER BY name
    """
    return [r[0] for r in con.execute(q).fetchall()]

def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    q = f"PRAGMA table_info({table})"
    return [row[1] for row in con.execute(q).fetchall()]

def pick_incremental_col(cols: List[str], priority: Tuple[str, ...]) -> Optional[str]:
    cols_set = set(c.lower() for c in cols)
    for p in priority:
        if p.lower() in cols_set:
            # return the actual column name as stored (case-sensitive safe)
            for c in cols:
                if c.lower() == p.lower():
                    return c
    return None


def pick_incremental_col_with_fallback(con: sqlite3.Connection, table: str, cols: List[str], priority: Tuple[str, ...]) -> Optional[str]:
    """Pick an incremental column using priority, falling back to a table PK or rowid.

    This prevents full-table re-dumps when there is no obvious timestamp column.
    """
    # First use the configured priority
    inc = pick_incremental_col(cols, priority)
    if inc:
        return inc

    # Try to find a declared primary key
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        for r in rows:
            # r[5] is pk (0=no, >0 position in composite key)
            if r[5]:
                return r[1]
    except Exception:
        pass

    # Fall back to rowid if it exists
    try:
        con.execute(f"SELECT rowid FROM {table} LIMIT 1").fetchone()
        return "rowid"
    except sqlite3.OperationalError:
        return None

def normalize_iso(ts: str) -> str:
    # expects an ISO-ish string; store normalized ISO UTC when possible
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(dt):
        return ts
    return dt.to_pydatetime().replace(microsecond=0).isoformat()

def max_timestamp(df: pd.DataFrame, col: str) -> Optional[str]:
    if df.empty or col not in df.columns:
        return None
    # If column looks numeric (e.g., integer PK / rowid), return numeric max as string
    if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
        m = df[col].max(skipna=True)
        if pd.isna(m):
            return None
        return str(int(m)) if pd.api.types.is_integer_dtype(df[col]) else str(m)
    # Otherwise try to parse as datetime
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    if s.isna().all():
        return None
    return s.max().to_pydatetime().replace(microsecond=0).isoformat()

def read_incremental_query(
    table: str,
    inc_col: Optional[str],
    last_ts: Optional[str],
    backfill_days: int,
) -> Tuple[str, List[str]]:
    """
    Returns (sql, params). We use parameter binding.
    """
    if inc_col is None:
        return (f"SELECT * FROM {table}", [])

    clauses = []
    params: List[str] = []

    if last_ts:
        clauses.append(f"({inc_col} IS NOT NULL AND {inc_col} > ?)")
        params.append(last_ts)

    if backfill_days and backfill_days > 0:
        cutoff = (utc_now() - timedelta(days=backfill_days)).replace(microsecond=0).isoformat()
        clauses.append(f"({inc_col} IS NOT NULL AND {inc_col} >= ?)")
        params.append(cutoff)

    if clauses:
        where = " OR ".join(clauses)
        return (f"SELECT * FROM {table} WHERE {where}", params)

    # First run (no last_ts): export all
    return (f"SELECT * FROM {table}", [])

def pick_creation_col(cols: List[str], priority: Tuple[str, ...]) -> Optional[str]:
    """Pick the best column to use as the row 'creation' timestamp (dump_date source)."""
    cols_set = set(c.lower() for c in cols)
    for p in priority:
        if p.lower() in cols_set:
            for c in cols:
                if c.lower() == p.lower():
                    return c
    return None


def get_primary_key_cols(con: sqlite3.Connection, table: str, cols: List[str]) -> Optional[List[str]]:
    """Return declared PK columns as a list (in PK order) if present, else ['rowid'] if available, else None."""
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        # r[5] is pk position (0=no, >0 position in composite key)
        pk_with_pos = [(r[5], r[1]) for r in rows if r[5]]
        if pk_with_pos:
            pk_with_pos.sort()
            return [c for pos, c in pk_with_pos]
    except Exception:
        pass

    # Fall back to rowid
    try:
        con.execute(f"SELECT rowid FROM {table} LIMIT 1").fetchone()
        return ["rowid"]
    except sqlite3.OperationalError:
        return None


def derive_dump_dates(df: pd.DataFrame, creation_col: Optional[str]) -> pd.Series:
    """Derive a 'YYYY-MM-DD' dump_date per row from the creation column. Falls back to today."""
    if creation_col and creation_col in df.columns:
        s = pd.to_datetime(df[creation_col], errors="coerce", utc=True)
        # Try to coerce numeric unix timestamps (seconds)
        mask = s.isna()
        if mask.any():
            nu = pd.to_numeric(df.loc[mask, creation_col], errors="coerce")
            parsed = pd.to_datetime(nu, unit='s', errors='coerce', utc=True)
            s.loc[mask] = parsed
        dates = s.dt.date.astype(str)
        dates = dates.fillna(today_utc_date())
        return dates
    # No creation_col present: default to today's dump date
    return pd.Series([today_utc_date()] * len(df), index=df.index)


def write_parquet_by_creation_date(
    df: pd.DataFrame,
    out_dir: Path,
    table: str,
    pk_cols: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[Path]:
    """Write one parquet file per unique dump_date found in df. Merge with existing files and deduplicate by PK columns if available."""
    written: List[Path] = []
    if df.empty:
        return written

    df = df.copy()
    if "dump_date" not in df.columns:
        df["dump_date"] = today_utc_date()

    folder = out_dir / "data" / table
    folder.mkdir(parents=True, exist_ok=True)

    for dump_date, sub in df.groupby("dump_date"):
        file_path = folder / f"{dump_date}.parquet"
        try:
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, sub], ignore_index=True)
            else:
                combined = sub

            # Deduplicate by PK columns if they exist in the combined frame
            if pk_cols and all(pc in combined.columns for pc in pk_cols):
                combined = combined.drop_duplicates(subset=pk_cols, keep="last")
            elif pk_cols and "rowid" in pk_cols and "rowid" in combined.columns:
                combined = combined.drop_duplicates(subset=["rowid"], keep="last")
            else:
                combined = combined.drop_duplicates()

            combined.to_parquet(file_path, index=False)
            written.append(file_path)
            if verbose:
                print(f"[export] wrote {file_path} ({len(combined)} rows, pk={pk_cols})")
        except Exception as e:
            # Log and continue
            if verbose:
                print(f"[export] failed writing {file_path}: {e}")
    return written


# ----------------------------
# Export logic
# ----------------------------

def export_sqlite_to_parquet(cfg: ExportConfig) -> List[Path]:
    if not cfg.db_path.exists():
        raise FileNotFoundError(f"DB not found: {cfg.db_path}")

    dump_date = today_utc_date()
    state = load_state(cfg.state_path)
    exported: List[Path] = []

    with sqlite3.connect(str(cfg.db_path)) as con:
        con.row_factory = sqlite3.Row

        tables = list_tables(con)

        # save a schema manifest for convenience
        manifest = {
            "db": str(cfg.db_path),
            "dump_date": dump_date,
            "tables": {},
        }

        for table in tables:
            cols = table_columns(con, table)
            inc_col = pick_incremental_col_with_fallback(con, table, cols, cfg.incremental_col_priority)
            creation_col = pick_creation_col(cols, cfg.creation_col_priority)
            pk_cols = get_primary_key_cols(con, table, cols)

            last_ts = state["last_exported"].get(table)
            backfill_days = cfg.per_table_backfill_days.get(table, cfg.default_backfill_days)

            if cfg.verbose:
                print(f"[export] table={table} inc_col={inc_col} creation_col={creation_col} pk={pk_cols} last_ts={last_ts} backfill_days={backfill_days}")

            sql, params = read_incremental_query(table, inc_col, last_ts, backfill_days)

            # If we require rowid as a primary key, ensure it's selected so we can deduplicate properly
            if pk_cols and pk_cols == ["rowid"] and "rowid" not in cols:
                # read_incremental_query produces SQL with SELECT * or SELECT * ... WHERE ...
                sql = sql.replace("SELECT *", "SELECT rowid, *", 1)

            # Chunked reading if large (LIMIT/OFFSET is okay for most SQLite use cases)
            # We read in chunks but concatenate them to produce per-creation-date parquet files for the table.
            df_iter = pd.read_sql_query(sql, con, params=params, chunksize=cfg.chunk_rows) 

            table_max_ts = None
            parts_written = 0
            chunks = []

            for chunk in df_iter:
                if not chunk.empty:
                    chunks.append(chunk)
                    if inc_col:
                        chunk_max = max_timestamp(chunk, inc_col)
                        if chunk_max:
                            table_max_ts = chunk_max if table_max_ts is None else max(table_max_ts, chunk_max)

            # If we have data, assign per-row dump_date (creation date) and write one file per date
            if chunks:
                df_all = pd.concat(chunks, ignore_index=True)
                # derive per-row dump_date from creation_col (falls back to today)
                df_all["dump_date"] = derive_dump_dates(df_all, creation_col)

                written_files = write_parquet_by_creation_date(
                    df_all, cfg.out_dir, table, pk_cols=pk_cols, verbose=cfg.verbose
                )

                if written_files:
                    exported.extend(written_files)
                    parts_written = len(written_files)

            # Update state if we actually exported something and have an incremental col
            if inc_col and table_max_ts and parts_written > 0:
                # If this looks like a datetime, normalize to ISO UTC; otherwise keep numeric PK/rowid as-is
                maybe_dt = pd.to_datetime(table_max_ts, errors="coerce", utc=True)
                if not pd.isna(maybe_dt):
                    state_val = maybe_dt.to_pydatetime().replace(microsecond=0).isoformat()
                else:
                    state_val = table_max_ts
                state["last_exported"][table] = state_val
                if cfg.verbose:
                    print(f"[export] updated state for {table}: {state_val}")

            manifest["tables"][table] = {
                "columns": cols,
                "incremental_col": inc_col,
                "creation_col": creation_col,
                "primary_key": pk_cols,
                "last_exported_ts": state["last_exported"].get(table),
                "backfill_days": backfill_days,
                "parts_written": parts_written,
            }

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    save_state(cfg.state_path, state)
    return exported


# ----------------------------
# Optional: Push to Hugging Face
# ----------------------------

def push_folder_to_hf(cfg: ExportConfig, commit_message: str) -> None:
    if not cfg.hf_repo_id:
        raise ValueError("hf_repo_id is not set.")
    if HfApi is None:
        raise RuntimeError("huggingface_hub is not installed. pip install huggingface_hub")

    api = HfApi()
    create_repo(repo_id=cfg.hf_repo_id, repo_type=cfg.hf_repo_type, exist_ok=True)

    api.upload_folder(
        repo_id=cfg.hf_repo_id,
        repo_type=cfg.hf_repo_type,
        folder_path=str(cfg.out_dir),
        path_in_repo=".",
        commit_message=commit_message,
    )


# ----------------------------
# CLI entry
# ----------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Export SQLite tables to date-partitioned Parquet (incremental).")
    p.add_argument("--db", required=True, help="Path to SQLite DB file")
    p.add_argument("--out", default="hf_export", help="Output dir")
    p.add_argument("--hf_repo", default=None, help="Optional: HF dataset repo id: username/repo")
    p.add_argument("--push", action="store_true", help="Push output folder to HF")
    p.add_argument("--default_backfill_days", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    cfg = ExportConfig(
        db_path=Path(args.db),
        out_dir=Path(args.out),
        hf_repo_id=args.hf_repo,
        default_backfill_days=args.default_backfill_days,
        verbose=args.verbose,
    )

    exported = export_sqlite_to_parquet(cfg)
    print(f"Exported {len(exported)} parquet part files into {cfg.out_dir}")

    if args.push:
        msg = f"Incremental export {today_utc_date()} ({len(exported)} parquet files)"
        push_folder_to_hf(cfg, msg)
        print("Pushed to Hugging Face.")
