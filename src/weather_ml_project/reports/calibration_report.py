"""
Compute and export per-region calibration coefficients.

Compares terrain station data (ground truth) against satellite/reanalysis
sources (Open-Meteo, ERA5, NASA POWER) and produces:

  reports/calibration_table.csv      — summary table, one row per region
  reports/calibration_monthly.csv    — monthly breakdown (seasonal bias)
  reports/figures/calibration_bias.png — heatmap of additive biases
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


# ── Source definitions ────────────────────────────────────────────────────────
# (terrain_col, satellite_col, label)
TEMP_PAIRS = [
    ("terrain_temperature", "om_temperature",   "OM"),
    ("terrain_temperature", "era5_temperature",  "ERA5"),
    ("terrain_temperature", "nasa_temperature",  "NASA"),
]

PRECIP_PAIRS = [
    ("terrain_precipitation", "om_precipitation",   "OM"),
    ("terrain_precipitation", "era5_precipitation",  "ERA5"),
    ("terrain_precipitation", "nasa_precipitation",  "NASA"),
    ("terrain_precipitation", "chirps_precipitation","CHIRPS"),
]


# ── Core statistics ───────────────────────────────────────────────────────────

def _pair_stats(
    a: pd.Series,
    b: pd.Series,
) -> dict:
    """Return stats for a valid (terrain, satellite) pair of observations."""
    mask = a.notna() & b.notna() & (a != 0) & (b != 0)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 10:
        return {"n": n, "bias": np.nan, "rmse": np.nan, "r": np.nan,
                "add_correction": np.nan, "mul_factor": np.nan}

    bias      = float((a - b).mean())          # terrain - satellite
    rmse      = float(np.sqrt(((a - b) ** 2).mean()))
    r         = float(a.corr(b)) if n >= 2 else np.nan
    add_corr  = bias                            # satellite + add_corr ≈ terrain
    mul_fact  = float(a.mean() / b.mean()) if b.mean() != 0 else np.nan
    return {"n": n, "bias": round(bias, 4), "rmse": round(rmse, 4),
            "r": round(r, 4),
            "add_correction": round(add_corr, 4),
            "mul_factor": round(mul_fact, 4)}


def build_calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (region_id, variable, source) with calibration metrics."""
    rows = []

    lat_col = next((c for c in ["lat", "latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude"] if c in df.columns), None)

    for rid, g in df.groupby("region_id"):
        lat = float(g[lat_col].mean()) if lat_col else np.nan
        lon = float(g[lon_col].mean()) if lon_col else np.nan

        for terrain_col, sat_col, source in TEMP_PAIRS:
            if terrain_col not in g or sat_col not in g:
                continue
            s = _pair_stats(g[terrain_col], g[sat_col])
            rows.append({
                "region_id": rid, "lat": lat, "lon": lon,
                "variable": "temperature_C",
                "source": source,
                **s,
            })

        for terrain_col, sat_col, source in PRECIP_PAIRS:
            if terrain_col not in g or sat_col not in g:
                continue
            s = _pair_stats(g[terrain_col], g[sat_col])
            rows.append({
                "region_id": rid, "lat": lat, "lon": lon,
                "variable": "precipitation_mm",
                "source": source,
                **s,
            })

    cols = ["region_id", "lat", "lon", "variable", "source",
            "n", "bias", "rmse", "r", "add_correction", "mul_factor"]
    return pd.DataFrame(rows, columns=cols)


def build_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly bias per (region_id, month, source) for temperature."""
    rows = []
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month

    for rid, g in df.groupby("region_id"):
        for month, gm in g.groupby("month"):
            for terrain_col, sat_col, source in TEMP_PAIRS:
                if terrain_col not in gm or sat_col not in gm:
                    continue
                s = _pair_stats(gm[terrain_col], gm[sat_col])
                rows.append({
                    "region_id": rid, "month": month,
                    "source": source,
                    "bias": s["bias"], "rmse": s["rmse"], "n": s["n"],
                })

    return pd.DataFrame(rows)


def plot_bias_heatmap(table: pd.DataFrame, out_path: Path) -> None:
    """Heatmap — rows = regions, cols = sources, values = temperature bias."""
    temp = table[table["variable"] == "temperature_C"].copy()
    if temp.empty:
        return

    pivot = temp.pivot_table(
        index="region_id", columns="source", values="bias", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), max(5, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
    plt.colorbar(im, ax=ax, label="Biais additif (terrain − satellite) °C")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Biais température par région et source satellite\n"
                 "(rouge = satellite sous-estime terrain, bleu = sur-estime)",
                 fontsize=11)

    # Annotate cells with bias value
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 2.5 else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_calibration_report(
    df: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Compute all calibration metrics and save reports. Returns summary table."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[calibration] Calcul des biais par region et source…", flush=True)
    table = build_calibration_table(df)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    out_csv = reports_dir / "calibration_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"[calibration] -> {out_csv}  ({len(table)} lignes)", flush=True)

    # ── Compact readable view: pivot bias only ────────────────────────────────
    for var in ("temperature_C", "precipitation_mm"):
        sub = table[table["variable"] == var]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index=["region_id", "lat", "lon"],
            columns="source",
            values=["bias", "rmse", "r", "add_correction", "mul_factor"],
        )
        pivot.columns = [f"{stat}_{src}" for stat, src in pivot.columns]
        pivot = pivot.reset_index().round(4)
        fname = f"calibration_{var.split('_')[0]}_pivot.csv"
        pivot.to_csv(reports_dir / fname, index=False)
        print(f"[calibration] -> {reports_dir / fname}", flush=True)

    # ── Monthly table ─────────────────────────────────────────────────────────
    monthly = build_monthly_table(df)
    monthly.to_csv(reports_dir / "calibration_monthly.csv", index=False)
    print(f"[calibration] -> {reports_dir / 'calibration_monthly.csv'}", flush=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    heatmap_path = figures_dir / "calibration_bias_heatmap.png"
    plot_bias_heatmap(table, heatmap_path)
    print(f"[calibration] -> {heatmap_path}", flush=True)

    # ── Console summary ───────────────────────────────────────────────────────
    temp_table = table[table["variable"] == "temperature_C"]
    print("\n" + "=" * 65)
    print(f"{'BIAIS TEMPERATURE (terrain - satellite) en °C':^65}")
    print("=" * 65)
    print(f"{'Region':<8} {'Source':<8} {'N':>6} {'Biais':>7} {'RMSE':>7} "
          f"{'R':>6} {'Add.Corr':>9} {'MulFact':>8}")
    print("-" * 65)
    for _, row in temp_table.sort_values(["region_id","source"]).iterrows():
        if np.isnan(row["bias"]):
            continue
        print(f"{row['region_id']:<8} {row['source']:<8} {int(row['n']):>6} "
              f"{row['bias']:>+7.3f} {row['rmse']:>7.3f} {row['r']:>6.3f} "
              f"{row['add_correction']:>+9.3f} {row['mul_factor']:>8.4f}")
    print("=" * 65 + "\n")

    return table


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    cleaned_path = Path("data_processed/cleaned_data.csv")
    if not cleaned_path.exists():
        print(f"ERREUR: {cleaned_path} introuvable. Lancez d'abord python main.py")
        sys.exit(1)

    df = pd.read_csv(cleaned_path)
    df["date"] = pd.to_datetime(df["date"])
    run_calibration_report(df)
