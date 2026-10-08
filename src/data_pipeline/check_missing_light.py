
import argparse
from pathlib import Path
 
import pandas as pd
 
 
def get_site(path: Path) -> str:
    """Work out whether a file belongs to hospital set A or B.
 
    Uses the folder name if it contains 'setA'/'setB' (the PhysioNet layout).
    Otherwise falls back to the file name: A patients are p0xxxxx,
    B patients are p1xxxxx.
    """
    parts = [p.lower() for p in path.parts]
    if any("setb" in p for p in parts):
        return "B"
    if any("seta" in p for p in parts):
        return "A"
    return "A" if path.stem[1] == "0" else "B"
 
 
def load_patients(data_dir: Path, limit: int | None) -> pd.DataFrame:
    """Read every .psv file (pipe-delimited) into one combined dataframe."""
    files = sorted(data_dir.rglob("*.psv"))
    if not files:
        raise SystemExit(f"No .psv files found under {data_dir.resolve()}")
 
    if limit:
        # Take half from each site so a small test still covers both sets.
        a = [f for f in files if get_site(f) == "A"][: limit // 2]
        b = [f for f in files if get_site(f) == "B"][: limit - limit // 2]
        files = a + b
 
    print(f"Loading {len(files)} patient files...")
    frames = []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, sep="|")  # files are pipe-delimited, not comma
        df["patient_id"] = f.stem
        df["site"] = get_site(f)
        frames.append(df)
        if i % 5000 == 0:
            print(f"  ...{i} files read")
    return pd.concat(frames, ignore_index=True)
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="folder with .psv files")
    parser.add_argument("--limit", type=int, default=None, help="only load N files (for testing)")
    parser.add_argument("--out", default="data/processed/missing_report.csv")
    args = parser.parse_args()
 
    df = load_patients(Path(args.data_dir), args.limit)
 
    # Basic counts 
    n_patients = df["patient_id"].nunique()
    print(f"\nPatients: {n_patients}  (expected 40,336 for the full A + B set)")
    print(df.groupby("site")["patient_id"].nunique().to_string())
    print(f"Rows: {len(df)}")
 
    # Class imbalance: patient-level vs row-level 
    # Patient-level: share of patients who EVER have SepsisLabel = 1 (~7.3%).
    # Row-level: share of hourly rows labeled 1 (much lower).
    ever_septic = df.groupby("patient_id")["SepsisLabel"].max()
    print(f"\nSepsis-positive patients: {ever_septic.mean():.2%}")
    print(f"Sepsis-positive rows:     {df['SepsisLabel'].mean():.2%}")
 
    # Missing-value report 
    id_cols = ["patient_id", "site"]
    cols = [c for c in df.columns if c not in id_cols]
 
    report = pd.DataFrame(
        {
            "missing_pct_overall": df[cols].isna().mean() * 100,
            "missing_pct_siteA": df[df["site"] == "A"][cols].isna().mean() * 100,
            "missing_pct_siteB": df[df["site"] == "B"][cols].isna().mean() * 100,
            # For each patient: was this column ever non-missing? Then average.
            "patients_ever_measured_pct": df.groupby("patient_id")[cols]
            .apply(lambda g: g.notna().any())
            .mean()
            * 100,
        }
    ).round(1)
    report = report.sort_values("missing_pct_overall", ascending=False)
 
    pd.set_option("display.max_rows", None, "display.width", 140)
    print("\nMissing-value report (sorted, most missing first):\n")
    print(report.to_string())
 
    # Flag candidates to drop 
    # Candidates: columns that are nearly always empty AND rarely measured for
    # any patient. 
    row_cutoff, patient_cutoff = 95, 30
    drop = report[
        (report["missing_pct_overall"] > row_cutoff)
        & (report["patients_ever_measured_pct"] < patient_cutoff)
    ]
    print(
        f"\nDrop candidates (>{row_cutoff}% rows missing AND measured for "
        f"<{patient_cutoff}% of patients):"
    )
    print(list(drop.index) if len(drop) else "none")
 
    gap = (report["missing_pct_siteA"] - report["missing_pct_siteB"]).abs()
    big = gap[gap > 20].sort_values(ascending=False)
    if len(big):
        print("\nColumns whose missing % differs by >20 points between sites:")
        print(big.round(1).to_string())
 
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out)
    print(f"\nSaved report to {out}")
 
 
if __name__ == "__main__":
    main()
 




