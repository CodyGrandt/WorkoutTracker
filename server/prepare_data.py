"""Download the OpenPowerlifting dataset and reduce it to one Raw bench-press baseline per lifter.

The ~170 MB zip is cached in the system temp directory (not the repo); only the
filtered CSV (server/data/opl_raw_bench.csv) is written to the project.
"""
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
ZIP_PATH = Path(tempfile.gettempdir()) / "openpowerlifting-latest.zip"
OUT_PATH = Path(__file__).parent / "data" / "opl_raw_bench.csv"
COLUMNS = ["Name", "Sex", "Equipment", "Age", "BodyweightKg", "Best3BenchKg", "Date"]


def download() -> None:
    if ZIP_PATH.exists():
        print(f"Using cached {ZIP_PATH}")
        return
    print(f"Downloading {URL} -> {ZIP_PATH} (~170 MB)...")
    urllib.request.urlretrieve(URL, ZIP_PATH)


def main() -> None:
    download()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        print(f"Reading {csv_name}")
        with zf.open(csv_name) as f:
            raw = pd.read_csv(f, usecols=COLUMNS, low_memory=False)
    print(f"Total competition entries: {len(raw):,}")

    df = raw[(raw["Equipment"] == "Raw") & raw["Sex"].isin(["M", "F"])]
    df = df.dropna(subset=["Age", "BodyweightKg", "Best3BenchKg"])
    # Failed attempts are stored as negative numbers; keep plausible successful benches only.
    df = df[(df["Best3BenchKg"] >= 20) & (df["Age"].between(16, 80)) & (df["BodyweightKg"].between(40, 200))]
    # One baseline per lifter: their most recent qualifying meet.
    df = df.sort_values("Date", kind="stable").drop_duplicates("Name", keep="last")

    out = pd.DataFrame(
        {
            "bodyweight_kg": df["BodyweightKg"],
            "age": df["Age"],
            "sex_binary": (df["Sex"] == "M").astype(int),
            "bench_1rm_baseline": df["Best3BenchKg"],
        }
    )
    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Raw, bench-tested unique lifters: {len(out):,} -> {OUT_PATH}")
    print(out.describe().round(1).to_string())


if __name__ == "__main__":
    main()
