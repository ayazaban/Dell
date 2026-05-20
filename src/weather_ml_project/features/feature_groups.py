"""Feature group definitions by satellite data source."""
from __future__ import annotations

SOURCES = ["era5", "openmeteo", "nasa", "fusion"]

SOURCE_LABELS: dict[str, str] = {
    "era5":      "ERA5",
    "openmeteo": "Open-Meteo",
    "nasa":      "NASA POWER",
    "fusion":    "Fusion (tous)",
}

_SOURCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "era5":      ("era5_",),
    "openmeteo": ("om_",),
    "nasa":      ("nasa_",),
    "fusion":    ("era5_", "om_", "nasa_", "chirps_"),
}

_TEMPORAL_COLS = frozenset({
    "month", "dayofyear", "dayofweek", "is_weekend",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "lat", "lon", "latitude", "longitude", "altitude", "year",
})

_DERIVED_COLS = frozenset({
    "terrain_vpd", "terrain_temp_anomaly", "om_temp_amplitude",
})


def get_features_for_source(
    source: str,
    all_cols: list[str],
    exclude: set[str] | None = None,
) -> list[str]:
    """Return feature columns for a given satellite source.

    Always includes:
      - Satellite columns with the source prefix(es)
      - Terrain lag columns (terrain_*_t-N) — never raw terrain target columns
      - Temporal/cyclic features
      - Derived features (VPD, amplitude, anomaly)
    """
    if source not in _SOURCE_PREFIXES:
        raise ValueError(f"Unknown source {source!r}. Valid: {SOURCES}")

    excl = set(exclude) if exclude else set()
    prefixes = _SOURCE_PREFIXES[source]

    result: list[str] = []
    for col in all_cols:
        if col in excl:
            continue
        if any(col.startswith(p) for p in prefixes):
            result.append(col)
        elif col.startswith("terrain_") and "_t-" in col:
            result.append(col)
        elif col in _DERIVED_COLS:
            result.append(col)
        elif col in _TEMPORAL_COLS:
            result.append(col)

    return result
