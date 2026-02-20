import pandas as pd
import os
import glob
import sys
import argparse
from typing import Any, Dict, List, Optional

def _series_mean_as_percent(series: pd.Series) -> float:
    """
    Robustly compute mean(0/1) * 100 for columns that may be bool/0-1/True-False strings.
    """
    if series is None or len(series) == 0:
        return 0.0

    s = series
    # bool dtype
    if pd.api.types.is_bool_dtype(s):
        return float(s.mean() * 100.0)

    # numeric dtype
    if pd.api.types.is_numeric_dtype(s):
        return float(s.mean() * 100.0)

    # strings / objects: map common truthy/falsy to 1/0
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
                "t": 1,
                "f": 0,
            }
        )
    )
    mapped = mapped.dropna()
    if len(mapped) == 0:
        return 0.0
    return float(mapped.mean() * 100.0)


def _env_tag_from_names(db_name: str, query_name: str) -> str:
    """
    Environment tag based on leading character(s) of sequence names.
    - 'K' -> Karawatha
    - 'V' -> Venman
    - otherwise 'Other'
    For inter comparisons across environments, returns 'Mixed'.
    """
    db_tag = (db_name or "")[:1].upper()
    q_tag = (query_name or "")[:1].upper()
    if db_tag and q_tag and db_tag != q_tag:
        return "Mixed"
    tag = db_tag or q_tag
    if tag == "K":
        return "Karawatha"
    if tag == "V":
        return "Venman"
    return "Other"


def _parse_result_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Returns a dict describing the result row or None if unrecognized.
    Expected filenames:
    - Inter: viz_frames_{QUERY}_vs_{DB}.csv (optionally *_cleaned.csv)
    - Intra: viz_intra_frames_{SEQ}.csv
    """
    if "summary_table" in filename:
        return None

    # Inter filenames follow official WildCross_VPR convention: viz_frames_{QUERY}_vs_{DB}.csv
    if "_vs_" in filename:
        clean_name = (
            filename.replace("viz_frames_", "")
            .replace("_cleaned.csv", "")
            .replace(".csv", "")
        )
        parts = clean_name.split("_vs_")
        if len(parts) != 2:
            return None
        query_name, db_name = parts[0], parts[1]
        return {
            "Type": "Inter",
            "Database": db_name,
            "Query": query_name,
            "Env": _env_tag_from_names(db_name=db_name, query_name=query_name),
        }

    # Intra
    if "intra_frames_" in filename:
        seq_name = filename.replace("viz_intra_frames_", "").replace(".csv", "")
        return {
            "Type": "Intra",
            "Sequence": seq_name,
            "Env": _env_tag_from_names(db_name=seq_name, query_name=seq_name),
        }

    return None


def _collect_results(results_dirs: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    processed_ids = set()

    for results_dir in results_dirs:
        all_files = glob.glob(os.path.join(results_dir, "*.csv"))
        all_files.sort()
        all_files_set = set(all_files)

        for fpath in all_files:
            filename = os.path.basename(fpath)
            meta = _parse_result_filename(filename)
            if meta is None:
                continue

            # Prefer *_cleaned.csv if present
            cleaned_path = fpath.replace(".csv", "_cleaned.csv")
            if not fpath.endswith("_cleaned.csv") and cleaned_path in all_files_set:
                continue

            # Dedup key (in case multiple dirs contain same logical result)
            if meta["Type"] == "Inter":
                rid = f"Inter:{meta['Database']}:{meta['Query']}"
            else:
                rid = f"Intra:{meta['Sequence']}"
            if rid in processed_ids:
                continue

            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue

            if len(df) == 0:
                r1 = 0.0
                r5 = 0.0
            else:
                r1 = _series_mean_as_percent(df["success"]) if "success" in df.columns else 0.0
                r5 = _series_mean_as_percent(df["success_r5"]) if "success_r5" in df.columns else 0.0

            row: Dict[str, Any] = {
                **meta,
                "File": os.path.relpath(fpath),
                "R1": float(r1),
                "R5": float(r5),
                "N": int(len(df)),
            }
            rows.append(row)
            processed_ids.add(rid)

    return rows


def _print_env_averages(rows: List[Dict[str, Any]], type_name: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\nNo {type_name} results for environment averages.")
        return

    print("\n" + "=" * 34)
    print(f" {type_name.upper()} ENVIRONMENT AVERAGES")
    print("=" * 34)
    for env in ["Karawatha", "Venman", "Mixed", "Other"]:
        sub = df[df["Env"] == env]
        if len(sub) == 0:
            continue
        print(f"{env:9s}  Mean R@1: {sub['R1'].mean():.2f}%   Mean R@5: {sub['R5'].mean():.2f}%   (n={len(sub)})")
    print("=" * 34)

def _print_inter_env_averages_wildcross_style(df_inter: pd.DataFrame) -> None:
    """
    Match WildCross-Replication inter averaging style:
      - compute a per-sequence metric first
      - then average across sequences (so sequences are equally weighted)

    For Pair-VPR inter results, each CSV corresponds to (Query sequence vs Database sequence).
    The closest per-sequence quantity is to aggregate by Query sequence across all its DB pairings.
    """
    if df_inter is None or df_inter.empty:
        print("\nNo Inter results for environment averages.")
        return

    # Per (Env, Query) mean across DB pairings/files (macro-average over DBs)
    per_query = (
        df_inter.groupby(["Env", "Query"], as_index=False)[["R1", "R5"]]
        .mean(numeric_only=True)
        .sort_values(["Env", "Query"])
    )

    print("\n" + "=" * 34)
    print(" INTER ENVIRONMENT AVERAGES (SEQUENCE-LEVEL)")
    print("=" * 34)
    for env in ["Karawatha", "Venman", "Mixed", "Other"]:
        sub = per_query[per_query["Env"] == env]
        if len(sub) == 0:
            continue
        print(
            f"{env:9s}  Mean R@1: {sub['R1'].mean():.2f}%   Mean R@5: {sub['R5'].mean():.2f}%   (n_seq={len(sub)})"
        )
    print("=" * 34)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Pair-VPR inter/intra results (R@1 and R@5).")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--inter_dir", default=os.path.join(base_dir, "logs", "csv_results"))
    parser.add_argument("--intra_dir", default=os.path.join(base_dir, "logs", "intra_results"))
    args = parser.parse_args()

    inter_dir = args.inter_dir
    intra_dir = args.intra_dir

    print("--- Generating summaries ---")
    print(f"- Inter dir: {inter_dir}")
    print(f"- Intra dir: {intra_dir}")

    rows = _collect_results([inter_dir, intra_dir])
    if not rows:
        print("No results found.")
        return

    df_all = pd.DataFrame(rows)
    pd.options.display.float_format = "{:,.2f}".format

    # -------------------------
    # INTER (raw + averaged)
    # -------------------------
    df_inter = df_all[df_all["Type"] == "Inter"].copy()
    if not df_inter.empty:
        df_inter = df_inter.sort_values(["Database", "Query"])
        print("\n" + "=" * 80)
        print(" INTER RESULTS (RAW)  -  R@1 and R@5")
        print("=" * 80)
        print(df_inter[["Database", "Query", "Env", "R1", "R5", "N"]].to_string(index=False))
        print("=" * 80)

        # Matrices (averaged view)
        m_r1 = df_inter.pivot(index="Database", columns="Query", values="R1").sort_index(axis=0).sort_index(axis=1)
        m_r5 = df_inter.pivot(index="Database", columns="Query", values="R5").sort_index(axis=0).sort_index(axis=1)
        m_r1["Row Avg"] = m_r1.mean(axis=1)
        m_r5["Row Avg"] = m_r5.mean(axis=1)

        print("\n" + "=" * 80)
        print(" INTER RESULTS (AVERAGED VIEW)  -  MATRIX R@1")
        print("=" * 80)
        print(m_r1.fillna("-").to_string())
        print("=" * 80)

        print("\n" + "=" * 80)
        print(" INTER RESULTS (AVERAGED VIEW)  -  MATRIX R@5")
        print("=" * 80)
        print(m_r5.fillna("-").to_string())
        print("=" * 80)

        # WildCross-style inter averaging: average per Query sequence first, then across sequences.
        per_query = df_inter.groupby(["Env", "Query"], as_index=False)[["R1", "R5"]].mean(numeric_only=True)
        print(
            f"\nInter overall mean (sequence-level): R@1={per_query['R1'].mean():.2f}%  R@5={per_query['R5'].mean():.2f}%  (n_seq={len(per_query)})"
        )
        _print_inter_env_averages_wildcross_style(df_inter)

        # Save summary CSVs
        m_r1.to_csv(os.path.join(base_dir, "wildcross_inter_summary_R1.csv"))
        m_r5.to_csv(os.path.join(base_dir, "wildcross_inter_summary_R5.csv"))
    else:
        print("\nNo inter results found.")

    # -------------------------
    # INTRA (raw + averaged)
    # -------------------------
    df_intra = df_all[df_all["Type"] == "Intra"].copy()
    if not df_intra.empty:
        df_intra = df_intra.sort_values(["Sequence"])
        print("\n" + "=" * 80)
        print(" INTRA RESULTS (RAW)  -  R@1 and R@5")
        print("=" * 80)
        print(df_intra[["Sequence", "Env", "R1", "R5", "N"]].to_string(index=False))
        print("=" * 80)

        print(f"\nIntra overall mean: R@1={df_intra['R1'].mean():.2f}%  R@5={df_intra['R5'].mean():.2f}%  (n={len(df_intra)})")
        _print_env_averages(df_intra.to_dict(orient="records"), type_name="Intra")

        # Save intra summary CSV
        df_intra.to_csv(os.path.join(base_dir, "wildcross_intra_summary_R1_R5.csv"), index=False)
    else:
        print("\nNo intra results found.")

if __name__ == "__main__":
    main()