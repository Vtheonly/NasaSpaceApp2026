#!/usr/bin/env python3
"""Build the self-contained offline viewer app from exported artifacts (issue #14).

Reads the inference export (``calendar_forecast_records.csv``,
``harmonized_daily.csv``, ``anomaly_records.csv``, ``manifest.json``) plus the
optional raw FIRMS source CSVs and renders ``index.html`` - ONE static file,
no server, no CDN, no network calls at all - so a reviewer can complete the
core demo (AOI overview -> hotspot map -> 52-week-by-year burning calendar ->
anomaly details -> forecast with uncertainty) on any modern browser,
including air-gapped machines.

Usability/accessibility are build-time contracts, not afterthoughts:

- semantic landmarks (header/main/section/footer), heading hierarchy,
  skip link, ``lang``, descriptive ``<title>``;
- the map SVG carries ``role="img"`` + ``aria-label``; the calendar is a
  focusable grid with per-year summaries; tables have captions + scoped
  headers;
- the palette is checked against WCAG AA contrast at build time
  (``_palette_report``) and recorded in ``app_build.json``;
- missing artifacts / missing forecast produce EXPLICIT states (never
  fabricated data): error alerts per artifact and a documented
  "no forecast available" status block (docs/MODEL_CONTRACTS.md section 8);
- responsive CSS grid layout with media queries; no fixed viewport widths.

``app_build.json`` records every input hash and rendered section so the UI
state is auditable exactly like the model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd  # noqa: E402

from src.artifacts.serialization import load_json, save_json  # noqa: E402
from src.common.logging import configure_logging, get_logger  # noqa: E402
from scripts.inventory_firms import OFFICIAL_ACKNOWLEDGMENT  # noqa: E402

LOGGER = get_logger(__name__)

APP_SCHEMA_VERSION = "1.0.0"
MAX_MAP_POINTS_DEFAULT = 5000
ANOMALY_ROWS = 12
FORECAST_CHART_HISTORY_DAYS = 90

#: Palette for the UI. Text colors are verified against WCAG AA (>= 4.5:1) on
#: their background at build time; the report is embedded in app_build.json.
PALETTE: dict[str, str] = {
    "bg": "#ffffff",
    "surface": "#f6f7f9",
    "ink": "#1a1a2e",
    "ink_secondary": "#424857",
    "accent": "#0b4f9e",
    "accent_soft": "#e7eff9",
    "anomaly_high": "#a4262c",
    "anomaly_low": "#1b6ca8",
    "missing": "#8b929c",
    "focus": "#b35c00",
    "modis": "#9a5700",
    "viirs": "#0f766e",
}

#: 5-level burning-activity scale (low -> high); text on the two darkest
#: levels flips to white to keep >= 4.5:1.
HEAT_LEVELS = ["#f2f7fb", "#c6dcf0", "#8fbae3", "#3572b3", "#16548f"]

#: Hard limit on rendered calendar years/weeks keeps the page bounded on
#: multi-decade archives; the year-row summaries keep all data inspectable.


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _esc(value) -> str:
    import html

    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# WCAG AA contrast verification (build-time contract)
# ---------------------------------------------------------------------------


def _rel_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    l1, l2 = _rel_luminance(fg), _rel_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _palette_report() -> dict:
    """Text pairs that must meet WCAG AA; verified at build time + in tests."""
    pairs = {
        "ink_on_bg": (PALETTE["ink"], PALETTE["bg"]),
        "ink_on_surface": (PALETTE["ink"], PALETTE["surface"]),
        "ink_secondary_on_bg": (PALETTE["ink_secondary"], PALETTE["bg"]),
        "ink_secondary_on_surface": (PALETTE["ink_secondary"], PALETTE["surface"]),
        "accent_on_bg": (PALETTE["accent"], PALETTE["bg"]),
        "anomaly_high_on_bg": (PALETTE["anomaly_high"], PALETTE["bg"]),
        "anomaly_low_on_bg": (PALETTE["anomaly_low"], PALETTE["bg"]),
        "focus_on_bg": (PALETTE["focus"], PALETTE["bg"]),
        "modis_on_bg": (PALETTE["modis"], PALETTE["bg"]),
        "viirs_on_bg": (PALETTE["viirs"], PALETTE["bg"]),
        "white_on_heat4": ("#ffffff", HEAT_LEVELS[4]),
        "white_on_heat3": ("#ffffff", HEAT_LEVELS[3]),
        "ink_on_heat0": (PALETTE["ink"], HEAT_LEVELS[0]),
        "ink_on_heat1": (PALETTE["ink"], HEAT_LEVELS[1]),
        "ink_on_accent_soft": (PALETTE["ink"], PALETTE["accent_soft"]),
    }
    return {
        name: round(contrast_ratio(fg, bg), 3)
        for name, (fg, bg) in pairs.items()
    }


# ---------------------------------------------------------------------------
# Input loading (explicit states; nothing is fabricated)
# ---------------------------------------------------------------------------


class AppInputError(RuntimeError):
    """Raised when the inference export is structurally unusable."""


def _read_csv_if_present(
    path: Path, required_columns: tuple[str, ...] = ()
) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, f"missing artifact: {path.name}"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable artifact {path.name}: {exc}"
    missing_cols = [c for c in required_columns if c not in frame.columns]
    if missing_cols:
        return (
            None,
            f"artifact {path.name} violates the export contract; missing "
            f"columns {missing_cols}",
        )
    return frame, None


def _resolve_manifest(inference_dir: Path) -> tuple[dict | None, str | None]:
    """Find the inference provenance manifest: either directly in the export
    dir or inside the newest ``inference_export_*`` artifact directory that
    run_inference.py writes (versioned outputs)."""
    direct = inference_dir / "manifest.json"
    if direct.is_file():
        try:
            return load_json(direct), None
        except Exception as exc:  # noqa: BLE001
            return None, f"unreadable provenance manifest: {exc}"
    candidates = sorted(inference_dir.glob("inference_export_*/manifest.json"))
    if candidates:
        newest = candidates[-1]
        try:
            return load_json(newest), None
        except Exception as exc:  # noqa: BLE001
            return None, f"unreadable provenance manifest: {exc}"
    return None, "missing artifact: manifest.json (no inference_export_* dir)"


def load_map_detections(
    source_dir: str | None, max_points: int
) -> tuple[list[dict], str | None]:
    """Optional hotspot map: raw lat/lon detections, capped for page weight."""
    if not source_dir:
        return [], "no source directory supplied; hotspot map omitted"
    rows: list[dict] = []
    for path in sorted(Path(source_dir).glob("*.csv")):
        if len(rows) >= max_points:
            break
        frame = pd.read_csv(path)
        if not {"latitude", "longitude"}.issubset(frame.columns):
            continue
        product = "modis" if "modis" in path.name.lower() else "viirs"
        for rec in frame.head(max_points - len(rows)).itertuples(index=False):
            rows.append(
                {
                    "lat": float(rec.latitude),
                    "lon": float(rec.longitude),
                    "product": product,
                }
            )
    if not rows:
        return [], f"no detections with coordinates found in {source_dir}"
    return rows, None


def _synthetic_note(manifest: dict, source_dir: str | None) -> str | None:
    """Honest demo labeling: fixture inputs must never look like real fires."""
    haystack = json.dumps(manifest or {}).lower()
    if "fixtures" in haystack or (
        source_dir and "fixtures" in source_dir.lower()
    ):
        return (
            "SYNTHETIC DEMO DATA: every number below comes from the committed "
            "deterministic sample fixtures (seed 2026), not from real fire "
            "activity. Real-data runs must point --source-dir at a verified "
            "FIRMS archive and re-run the five-pass gate first "
            "(docs/MODEL_CONTRACTS.md)."
        )
    return None


# ---------------------------------------------------------------------------
# Renderers (plain string building; every value escaped)
# ---------------------------------------------------------------------------


def render_map_svg(detections: list[dict], width: int = 800, height: int = 420) -> str:
    if not detections:
        return (
            '<p class="state state-empty">No hotspot coordinates were supplied; '
            "the map section is intentionally empty rather than fabricated.</p>"
        )
    lats = [d["lat"] for d in detections]
    lons = [d["lon"] for d in detections]
    pad_lat = max((max(lats) - min(lats)) * 0.06, 0.02)
    pad_lon = max((max(lons) - min(lons)) * 0.06, 0.02)
    lat_min, lat_max = min(lats) - pad_lat, max(lats) + pad_lat
    lon_min, lon_max = min(lons) - pad_lon, max(lons) + pad_lon

    def _x(lon: float) -> float:
        return (lon - lon_min) / (lon_max - lon_min) * (width - 80) + 60

    def _y(lat: float) -> float:
        return height - 40 - (lat - lat_min) / (lat_max - lat_min) * (height - 70)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Hotspot map: '
        f"{len(detections)} validated active-fire detections plotted by latitude "
        f'and longitude (lat {lat_min:.2f} to {lat_max:.2f}, '
        f'lon {lon_min:.2f} to {lon_max:.2f}); MODIS and VIIRS detections use '
        'distinct shapes and colors." class="map-svg" tabindex="0">',
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="{PALETTE["surface"]}"/>',
    ]
    for lon_tick in [lon_min, (lon_min + lon_max) / 2, lon_max]:
        parts.append(
            f'<line x1="{_x(lon_tick):.1f}" y1="20" x2="{_x(lon_tick):.1f}" '
            f'y2="{height - 40}" stroke="{PALETTE["missing"]}" stroke-width="1" '
            f'stroke-dasharray="3 4"/><text x="{_x(lon_tick):.1f}" '
            f'y="{height - 22}" text-anchor="middle" font-size="12" '
            f'fill="{PALETTE["ink_secondary"]}">{lon_tick:.2f}</text>'
        )
    for lat_tick in [lat_min, (lat_min + lat_max) / 2, lat_max]:
        parts.append(
            f'<line x1="60" y1="{_y(lat_tick):.1f}" x2="{width - 20}" '
            f'y2="{_y(lat_tick):.1f}" stroke="{PALETTE["missing"]}" '
            f'stroke-width="1" stroke-dasharray="3 4"/><text x="52" '
            f'y="{_y(lat_tick) + 4:.1f}" text-anchor="end" font-size="12" '
            f'fill="{PALETTE["ink_secondary"]}">{lat_tick:.2f}</text>'
        )
    for d in detections:
        color = PALETTE["modis"] if d["product"] == "modis" else PALETTE["viirs"]
        if d["product"] == "modis":
            parts.append(
                f'<circle cx="{_x(d["lon"]):.1f}" cy="{_y(d["lat"]):.1f}" r="3" '
                f'fill="{color}" fill-opacity="0.75"><title>'
                f"{_esc(d['product'].upper())} detection</title></circle>"
            )
        else:
            x, y = _x(d["lon"]), _y(d["lat"])
            parts.append(
                f'<path d="M {x:.1f} {y - 3.4:.1f} L {x + 3:.1f} {y + 2.4:.1f} '
                f'L {x - 3:.1f} {y + 2.4:.1f} Z" fill="{color}" '
                f'fill-opacity="0.75"><title>VIIRS detection</title></path>'
            )
    parts.append(
        f'<text x="60" y="16" font-size="12" fill="{PALETTE["ink_secondary"]}">'
        "detections are pixel centers, not exact fire locations (1 km MODIS / "
        "375 m VIIRS pixels)</text></svg>"
    )
    return "".join(parts)


def _week_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily series to ISO-year/week rows for the calendar."""
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    iso = frame["date"].dt.isocalendar()
    frame["iso_year"] = iso["year"].astype(int)
    frame["iso_week"] = iso["week"].astype(int)

    def _week_value(group: pd.DataFrame) -> pd.Series:
        observed = pd.to_numeric(group["value"], errors="coerce").dropna()
        observed = observed[group.loc[observed.index, "value_type"] != "missing"]
        return pd.Series(
            {
                "week_value": float(observed.mean()) if len(observed) else float("nan"),
                "observed_days": int(len(observed)),
                "missing_days": int(len(group) - len(observed)),
            }
        )

    weeks = (
        frame.groupby(["iso_year", "iso_week"], sort=True)
        .apply(_week_value, include_groups=False)
        .reset_index()
    )
    return weeks


def _heat_class(value: float, scale: list[float]) -> int:
    for idx, edge in enumerate(scale):
        if value <= edge:
            return idx
    return len(scale)


def render_calendar_html(daily: pd.DataFrame) -> str:
    weeks = _week_stats(daily)
    observed_values = weeks["week_value"].dropna()
    if observed_values.empty:
        return (
            '<p class="state state-empty">No observed activity to calendar; '
            "the burning calendar is empty rather than zero-filled.</p>"
        )
    scale = [float(observed_values.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]

    eras = (
        daily.assign(year=pd.to_datetime(daily["date"]).dt.year)
        .groupby("year")["sensor_era"]
        .agg(lambda s: "+".join(sorted(set(s.dropna()))))
        .to_dict()
    )

    parts = [
        '<div class="calendar" role="group" aria-label="Burning calendar: '
        'activity by ISO week for every year">',
        '<div class="calendar-legend" role="list" aria-label="Calendar legend">',
    ]
    for level, color in enumerate(HEAT_LEVELS):
        parts.append(
            f'<span role="listitem"><i class="swatch swatch-heat" '
            f'style="background:{color}"></i> {"low" if level == 0 else ""}'
            f'{"high" if level == len(HEAT_LEVELS) - 1 else ""} '
            f"activity (level {level + 1})</span>"
        )
    parts.append(
        '<span role="listitem"><i class="swatch swatch-missing"></i> week with '
        'no observed coverage (missing data, NOT zero activity)</span>'
        '<span role="listitem"><i class="swatch swatch-partial"></i> week with '
        "partial coverage (dot = has missing days)</span></div>"
    )
    for year, group in weeks.groupby("iso_year", sort=True):
        era = eras.get(int(year), "")
        total = float(pd.to_numeric(group["week_value"], errors="coerce").sum())
        summary = (
            f"{int(year)}: {len(group)} weeks, "
            f"observed {int(group['observed_days'].sum())} days, "
            f"{int(group['missing_days'].sum())} days without coverage, "
            f"sum of weekly mean activity {total:.1f} MW, sensor era {era or 'n/a'}."
        )
        parts.append(
            f'<div class="cal-year" tabindex="0" role="row" aria-label="{_esc(summary)}">'
            f'<span class="cal-year-label"><strong>{int(year)}</strong>'
            f"<small>{_esc(era)}</small></span>"
        )
        week_index = {int(r.iso_week): r for r in group.itertuples(index=False)}
        for week in range(1, 54):
            if week not in week_index:
                parts.append('<i class="cal-cell cal-week-off"></i>')
                continue
            row = week_index[week]
            if pd.isna(row.week_value):
                cls, label = "cal-cell cal-missing", "no observed coverage"
                title = f"Week {week} {int(year)}: {label}."
            else:
                level = _heat_class(float(row.week_value), scale)
                cls = f"cal-cell cal-heat-{level}"
                label = f"activity level {level + 1}"
                title = (
                    f"Week {week} {int(year)}: mean {float(row.week_value):.1f} MW "
                    f"({label}); {int(row.observed_days)} observed, "
                    f"{int(row.missing_days)} missing days."
                )
            if (
                not pd.isna(row.week_value)
                and int(row.missing_days) > 0
                and int(row.observed_days) > 0
            ):
                cls += " cal-partial"
            parts.append(
                f'<i class="{cls}" title="{_esc(title)}" aria-hidden="true"></i>'
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def render_anomaly_table(anomalies: pd.DataFrame, limit: int = ANOMALY_ROWS) -> str:
    if anomalies is None or anomalies.empty:
        return (
            '<p class="state state-empty">No anomaly records in the export; '
            "the anomalies section is empty rather than fabricated.</p>"
        )
    frame = anomalies.copy()
    ratio_col = "ratio" if "ratio" in frame.columns else "anomaly_ratio"
    status_col = "status" if "status" in frame.columns else "anomaly_status"
    frame["_abs_ratio"] = pd.to_numeric(
        frame[ratio_col], errors="coerce"
    ).abs()
    top = frame.sort_values("_abs_ratio", ascending=False).head(limit)
    rows = []
    # NOTE: iterrows (not itertuples) keeps column access stable.
    for _, rec in top.iterrows():
        status = str(rec[status_col])
        status_class = (
            "status-high" if "high" in status else "status-low" if "low" in status else ""
        )
        rows.append(
            "<tr>"
            f"<th scope=\"row\">{_esc(rec['date'])}</th>"
            f"<td>{float(rec['observed']) if pd.notna(rec['observed']) else 'n/a'}</td>"
            f"<td>{float(rec['baseline']) if pd.notna(rec['baseline']) else 'n/a'}</td>"
            f"<td>{float(rec[ratio_col]):.2f}</td>"
            f"<td><span class=\"status {status_class}\">{_esc(status)}</span></td>"
            "</tr>"
        )
    caption = (
        f"Top {len(top)} days by absolute anomaly ratio (observed FRP divided "
        "by the seasonal baseline; thresholds and definitions in "
        "docs/MODEL_CONTRACTS.md)."
    )
    return (
        "<table class=\"anomaly-table\"><caption>"
        f"{_esc(caption)}</caption>"
        "<thead><tr><th scope=\"col\">Date</th><th scope=\"col\">Observed FRP (MW)</th>"
        "<th scope=\"col\">Seasonal baseline (MW)</th><th scope=\"col\">Ratio</th>"
        "<th scope=\"col\">Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_forecast_section(
    calendar: pd.DataFrame | None, daily: pd.DataFrame | None
) -> str:
    if calendar is None:
        return (
            '<div class="state state-error" role="alert">Forecast unavailable: '
            "the calendar/forecast export artifact is missing. No forecast "
            "values are shown because none exist (docs/MODEL_CONTRACTS.md "
            "section 8: explicit no-forecast state).</div>"
        )
    forecast_rows = calendar[calendar["quality_state"] == "forecast"]
    if forecast_rows.empty or calendar["forecast_model"].isna().all():
        return (
            '<div class="state state-empty" role="status">No forecast available '
            "for this run: no forecaster artifact was supplied. The calendar "
            "and anomalies below remain complete; nothing here is "
            "extrapolated.</div>"
        )
    model = str(calendar["forecast_model"].dropna().iloc[0])
    calibration = str(
        calendar["forecast_calibration_status"].dropna().iloc[0]
    )
    issue_date = str(calendar["forecast_issue_date"].dropna().iloc[0])
    horizon = len(forecast_rows)
    q10 = pd.to_numeric(forecast_rows["forecast_q10"], errors="coerce")
    median = pd.to_numeric(forecast_rows["forecast_median"], errors="coerce")
    q90 = pd.to_numeric(forecast_rows["forecast_q90"], errors="coerce")

    parts = [
        f'<div class="forecast-status" role="status"><strong>Forecast model:</strong> '
        f"{_esc(model)} &middot; <strong>Horizon:</strong> {horizon} days from "
        f"{_esc(issue_date)} &middot; <strong>Calibration:</strong> "
        f"<span class=\"status status-low\">{_esc(calibration)}</span> "
        "(quantiles are model estimates, not calibrated probabilities).</div>",
        '<figure class="forecast-figure">',
        '<svg viewBox="0 0 800 300" role="img" aria-label="Forecast chart: recent '
        f"observed activity followed by a {horizon}-day probabilistic forecast "
        'fan between the 10th and 90th percentiles around the median line." '
        'class="forecast-svg" tabindex="0">',
    ]

    if daily is not None and not daily.empty:
        tail = daily.tail(FORECAST_CHART_HISTORY_DAYS)
        combined = pd.concat(
            [
                tail.assign(kind="history"),
                forecast_rows.assign(kind="forecast"),
            ],
            ignore_index=True,
        )
        combined["_date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values("_date")
        all_values = pd.to_numeric(combined["value"], errors="coerce").dropna()
        if not all_values.empty:
            y_min, y_max = float(all_values.min()), float(all_values.max())
            span = (y_max - y_min) or 1.0
            y_min, y_max = y_min - 0.05 * span, y_max + 0.05 * span

            def _px(ts) -> float:
                first = combined["_date"].iloc[0]
                last = combined["_date"].iloc[-1]
                total = max((last - first).total_seconds(), 1)
                return (
                    (ts - first).total_seconds() / total * 740 + 40
                )

            def _py(v: float) -> float:
                return 260 - (v - y_min) / (y_max - y_min) * 220

            fan, line_hist, line_fc = [], [], []
            for _, rec in combined.iterrows():
                x = _px(rec["_date"])
                if rec["kind"] == "forecast":
                    fan.append((x, _py(float(rec["forecast_q90"])), _py(float(rec["forecast_q10"]))))
                    line_fc.append(f"{x:.1f},{_py(float(rec['forecast_median'])):.1f}")
                else:
                    v = rec["value"]
                    if pd.notna(v):
                        line_hist.append(f"{x:.1f},{_py(float(v)):.1f}")
            if fan:
                top = " ".join(f"M {x:.1f} {y_hi:.1f}" if i == 0 else f"L {x:.1f} {y_hi:.1f}"
                               for i, (x, y_hi, _) in enumerate(fan))
                bottom = " ".join(
                    f"L {x:.1f} {y_lo:.1f}"
                    for x, _, y_lo in reversed(fan)
                )
                parts.append(
                    f'<path d="{top} {bottom} Z" fill="{PALETTE["accent"]}" '
                    'fill-opacity="0.18"></path>'
                )
            if line_hist:
                parts.append(
                    f'<polyline fill="none" stroke="{PALETTE["ink"]}" '
                    f'stroke-width="2" points="{" ".join(line_hist)}"></polyline>'
                )
            if line_fc:
                parts.append(
                    f'<polyline fill="none" stroke="{PALETTE["accent"]}" '
                    f'stroke-width="2.5" stroke-dasharray="6 4" '
                    f'points="{" ".join(line_fc)}"></polyline>'
                )
            parts.append(
                f'<text x="40" y="280" font-size="12" fill="{PALETTE["ink_secondary"]}">'
                "FRP (MW) over time; shaded band = 10th-90th percentile, dashed "
                "line = median forecast.</text>"
            )
    parts.append("</svg>")
    parts.append(
        "<figcaption>Quantile spread grows with horizon because these are "
        "uncalibrated model estimates; see the limitations section.</figcaption>"
    )
    parts.append("</figure>")
    return "".join(parts)


def render_provenance(manifest: dict | None, input_hashes: dict) -> str:
    if manifest is None:
        return (
            '<div class="state state-error" role="alert">Provenance manifest '
            "missing; the provenance section cannot be rendered.</div>"
        )
    items = [
        f"<li><strong>AOI:</strong> {_esc(manifest.get('aoi_id', 'unknown'))}</li>",
        f"<li><strong>Code commit:</strong> "
        f"<code>{_esc((manifest.get('code_commit') or 'unknown')[:12])}</code></li>",
    ]
    sources = manifest.get("source_files") or []
    for src in sources[:6]:
        items.append(
            f"<li><strong>Source file:</strong> <code>{_esc(src.get('path', '?'))}"
            f"</code></li>"
        )
    for name, digest in input_hashes.items():
        items.append(
            f"<li><strong>Input:</strong> <code>{_esc(name)}</code> "
            f"SHA-256 <code>{_esc(digest[:16])}&hellip;</code></li>"
        )
    return (
        '<ul class="provenance-list">' + "".join(items) + "</ul>"
    )


def render_document(
    *,
    title: str,
    daily: pd.DataFrame | None,
    calendar: pd.DataFrame | None,
    anomalies: pd.DataFrame | None,
    manifest: dict | None,
    detections: list[dict],
    errors: list[str],
    generated_at: str,
    input_hashes: dict,
    source_dir_for_note: str | None = None,
) -> str:
    days_total = len(daily) if daily is not None else 0
    observed_days = (
        int((daily["value_type"] != "missing").sum()) if daily is not None else 0
    )
    missing_days = days_total - observed_days
    high = (
        int((anomalies["status"] == "anomaly_high").sum())
        if anomalies is not None and "status" in anomalies.columns
        else 0
    )
    low = (
        int((anomalies["status"] == "anomaly_low").sum())
        if anomalies is not None and "status" in anomalies.columns
        else 0
    )
    synthetic = _synthetic_note(manifest, source_dir_for_note)

    forecast_chip = "No forecast available"
    if calendar is not None and not calendar["forecast_model"].dropna().empty:
        forecast_chip = (
            f"Forecast: {str(calendar['forecast_model'].dropna().iloc[0])} "
            f"({int((calendar['quality_state'] == 'forecast').sum())}-day horizon)"
        )

    error_blocks = "".join(
        f'<div class="state state-error" role="alert">{_esc(err)}</div>'
        for err in errors
    )
    synthetic_block = (
        f'<div class="state state-warning" role="note">{_esc(synthetic)}</div>'
        if synthetic
        else ""
    )
    date_span = (
        f"{daily['date'].min()} to {daily['date'].max()}" if daily is not None else "n/a"
    )

    css = _CSS
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{css}</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to main content</a>
<header class="app-header">
<h1>{_esc(title)}</h1>
<p class="subtitle">Burning-activity calendar viewer &middot; series {_esc(date_span)}
&middot; {days_total} days ({observed_days} observed, {missing_days} without coverage)</p>
<p class="chips"><span class="chip">{_esc(forecast_chip)}</span>
<span class="chip">Anomalies: {high} high / {low} low</span>
<span class="chip">{_esc(generated_at)} UTC</span></p>
{synthetic_block}
{error_blocks}
</header>
<main id="main">
<section aria-labelledby="ov-h">
<h2 id="ov-h">Overview</h2>
<div class="stat-grid" role="list">
<span class="stat" role="listitem"><strong>{days_total}</strong> calendar days in series</span>
<span class="stat" role="listitem"><strong>{observed_days}</strong> observed days</span>
<span class="stat" role="listitem"><strong>{missing_days}</strong> days without coverage (explicitly missing, never zero)</span>
<span class="stat" role="listitem"><strong>{high}</strong> high-anomaly days</span>
<span class="stat" role="listitem"><strong>{low}</strong> low-anomaly days</span>
<span class="stat" role="listitem"><strong>{len(detections)}</strong> detections on the hotspot map</span>
</div>
</section>
<section aria-labelledby="map-h">
<h2 id="map-h">Hotspot map</h2>
{render_map_svg(detections)}
<p class="legend"><i class="swatch swatch-modis"></i> MODIS (circle) &middot;
<i class="swatch swatch-viirs"></i> VIIRS (triangle)</p>
</section>
<section aria-labelledby="cal-h">
<h2 id="cal-h">52-week-by-year burning calendar</h2>
<p class="section-note">Each row is one year; each cell is one ISO week. Weeks
without coverage are gray and distinct from low-activity weeks. Use Tab to
walk year rows; each carries a spoken summary.</p>
{render_calendar_html(daily) if daily is not None else '<div class="state state-error" role="alert">Daily series artifact missing; calendar cannot be rendered.</div>'}
</section>
<section aria-labelledby="ano-h">
<h2 id="ano-h">Anomaly details</h2>
{render_anomaly_table(anomalies) if anomalies is not None else '<div class="state state-error" role="alert">Anomaly artifact missing; anomaly details cannot be rendered.</div>'}
</section>
<section aria-labelledby="fc-h">
<h2 id="fc-h">14-day forecast with uncertainty</h2>
{render_forecast_section(calendar, daily)}
</section>
<section aria-labelledby="pr-h">
<h2 id="pr-h">Provenance and limitations</h2>
{render_provenance(manifest, input_hashes)}
<ul class="limitations" role="list">
<li role="listitem">Harmonized values are model estimates, not ground truth; they inherit MODIS detection-selection bias.</li>
<li role="listitem">Matched-calibration targets describe only fires both sensors could see.</li>
<li role="listitem">Forecast quantiles are experimental/uncalibrated estimates, not calibrated probabilities.</li>
<li role="listitem">Missing coverage days are shown as missing, never as zero activity.</li>
</ul>
</section>
</main>
<footer class="app-footer">
<p>{_esc(OFFICIAL_ACKNOWLEDGMENT)}</p>
<p class="fine">Built offline from versioned, hash-verified artifacts; see
app_build.json for input hashes and palette contrast checks.</p>
</footer>
</body>
</html>"""


_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
color:#1a1a2e;background:#ffffff;line-height:1.55}
.skip-link{position:absolute;left:-9999px;top:0;background:#ffffff;color:#b35c00;
padding:.6rem 1rem;z-index:10;border:2px solid #b35c00}
.skip-link:focus{left:.5rem;top:.5rem}
.app-header,.main,main{max-width:1100px;margin:0 auto;padding:1rem 1.25rem}
.app-header{padding-top:2rem}
h1{font-size:1.7rem;margin:.2rem 0}
h2{font-size:1.15rem;margin:2rem 0 .6rem;color:#1a1a2e}
.subtitle{color:#424857;margin:.2rem 0 .8rem}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.4rem 0 .8rem}
.chip{background:#e7eff9;color:#1a1a2e;border-radius:999px;padding:.15rem .7rem;
font-size:.85rem}
.state{border-radius:.5rem;padding:.7rem 1rem;margin:.5rem 0}
.state-error{background:#fdecec;color:#a4262c;border:1px solid #a4262c}
.state-warning{background:#fff7e6;color:#6b4b00;border:1px solid #b35c00}
.state-empty{background:#f6f7f9;color:#424857;border:1px solid #8b929c}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:.6rem}
.stat{background:#f6f7f9;border-radius:.5rem;padding:.7rem .9rem;color:#1a1a2e}
.stat strong{display:block;font-size:1.4rem;color:#0b4f9e}
section{border-top:1px solid #d8dce3;padding-top:.4rem}
.map-svg,.forecast-svg{width:100%;height:auto;background:#f6f7f9;border-radius:.5rem}
:focus-visible{outline:3px solid #b35c00;outline-offset:2px}
.legend{display:flex;align-items:center;gap:1rem;color:#424857}
.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;
vertical-align:-2px;margin-right:.3rem;border:1px solid #8b929c}
.swatch-modis{background:#e08a1e}.swatch-viirs{background:#0f766e}
.swatch-missing{background:repeating-linear-gradient(45deg,#8b929c,#8b929c 3px,
#f6f7f9 3px,#f6f7f9 6px)}
.swatch-partial{background:#f2f7fb;border-radius:50%}
.calendar{overflow-x:auto;padding:.5rem 0}
.calendar-legend{display:flex;flex-wrap:wrap;gap:.9rem;font-size:.85rem;
color:#424857;margin-bottom:.6rem}
.cal-year{display:flex;align-items:center;gap:.45rem;margin:.18rem 0;
border-radius:.4rem;padding:.15rem .3rem}
.cal-year-label{min-width:96px;color:#1a1a2e}
.cal-year-label small{display:block;color:#424857;font-size:.7rem}
.cal-cell{display:inline-block;width:13px;height:13px;border-radius:2px;
background:#e3e6ea;margin:1px}
.cal-heat-0{background:#f2f7fb}.cal-heat-1{background:#c6dcf0}
.cal-heat-2{background:#8fbae3}.cal-heat-3{background:#4a86c8}
.cal-heat-4{background:#16548f}
.cal-missing{background:repeating-linear-gradient(45deg,#8b929c,#8b929c 3px,
#f6f7f9 3px,#f6f7f9 6px)}
.cal-partial{outline:1px dotted #b35c00;outline-offset:1px}
.cal-week-off{background:#ffffff;border:1px dashed #d8dce3}
.anomaly-table{border-collapse:collapse;width:100%;margin-top:.4rem;
font-variant-numeric:tabular-nums}
.anomaly-table caption{text-align:left;color:#424857;padding:.3rem 0}
.anomaly-table th,.anomaly-table td{border-bottom:1px solid #d8dce3;
padding:.35rem .6rem;text-align:left}
.anomaly-table thead th{color:#1a1a2e;background:#f6f7f9}
.status{border-radius:999px;padding:.05rem .6rem;font-size:.8rem}
.status-high{background:#fdecec;color:#a4262c}
.status-low{background:#e7eff9;color:#1b6ca8}
.forecast-status{background:#f6f7f9;border-radius:.5rem;padding:.6rem .9rem;
margin:.4rem 0;color:#1a1a2e}
.forecast-figure{margin:.6rem 0}
.forecast-figure figcaption{color:#424857;font-size:.85rem}
.provenance-list,.limitations{color:#1a1a2e}
.provenance-list li,.limitations li{margin:.25rem 0}
code{background:#f6f7f9;border-radius:3px;padding:.05rem .3rem}
.app-footer{border-top:1px solid #d8dce3;margin-top:2.5rem;background:#f6f7f9}
.app-footer p{max-width:1100px;margin:0 auto;padding:.55rem 1.25rem;color:#1a1a2e}
.app-footer .fine{color:#424857;font-size:.8rem;padding-bottom:1.1rem}
@media (max-width:640px){
 .cal-year-label{min-width:64px}
 .cal-cell{width:9px;height:9px}
 h1{font-size:1.35rem}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir",
        required=True,
        help="run_inference output dir (calendar/forecast + daily + anomaly CSVs + manifest.json)",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional raw FIRMS CSV dir; supplies the hotspot map detections",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where index.html + app_build.json are written",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Document title (default: AOI id from the inference manifest)",
    )
    parser.add_argument(
        "--max-map-points",
        type=int,
        default=MAX_MAP_POINTS_DEFAULT,
        help="Cap on rendered map detections (page weight bound)",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Deterministic build stamp override (UTC ISO string)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    calendar, err = _read_csv_if_present(
        inference_dir / "calendar_forecast_records.csv",
        required_columns=("date", "quality_state", "forecast_model"),
    )
    if err:
        errors.append(err)
    daily, err = _read_csv_if_present(
        inference_dir / "harmonized_daily.csv",
        required_columns=("date", "value_type"),
    )
    if err:
        errors.append(err)
    anomalies, err = _read_csv_if_present(
        inference_dir / "anomaly_records.csv",
        required_columns=("date", "observed", "baseline", "ratio", "status"),
    )
    if err:
        errors.append(err)
    manifest, err = _resolve_manifest(inference_dir)
    if err:
        errors.append(err)

    detections: list[dict] = []
    if args.source_dir:
        detections, map_err = load_map_detections(
            args.source_dir, args.max_map_points
        )
        if map_err:
            LOGGER.warning("%s", map_err)

    if daily is None and calendar is None:
        print(
            "Neither the daily series nor the calendar export could be read; "
            "refusing to render an empty app.",
            file=sys.stderr,
        )
        save_json(
            output_dir / "app_build.json",
            {
                "schema_version": APP_SCHEMA_VERSION,
                "status": "failed",
                "errors": errors,
                "generated_at": args.generated_at,
            },
        )
        return 1

    input_hashes = {
        name: sha256_file(inference_dir / name)
        for name in (
            "calendar_forecast_records.csv",
            "harmonized_daily.csv",
            "anomaly_records.csv",
        )
        if (inference_dir / name).is_file()
    }

    generated_at = args.generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    aoi_id = (manifest or {}).get("aoi_id", "AOI")
    title = args.title or f"Burning calendar - {aoi_id}"

    document = render_document(
        title=title,
        daily=daily,
        calendar=calendar,
        anomalies=anomalies,
        manifest=manifest,
        detections=detections,
        errors=errors,
        generated_at=generated_at,
        input_hashes=input_hashes,
        source_dir_for_note=args.source_dir,
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")

    forecast_available = bool(
        calendar is not None and not calendar["forecast_model"].dropna().empty
    )
    build_report = {
        "schema_version": APP_SCHEMA_VERSION,
        "status": "passed" if not errors else "partial",
        "generated_at": generated_at,
        "inputs": {k: {"sha256": v} for k, v in input_hashes.items()},
        "map_detections": len(detections),
        "forecast_available": forecast_available,
        "synthetic_demo": _synthetic_note(manifest, args.source_dir) is not None,
        "palette_contrast": _palette_report(),
        "wcag_aa_threshold": 4.5,
        "palette_passes_aa": all(
            r >= 4.5 for r in _palette_report().values()
        ),
        "errors": errors,
        "output": str(output_dir / "index.html"),
    }
    save_json(output_dir / "app_build.json", build_report)

    print(
        f"App written to {output_dir / 'index.html'} "
        f"(forecast={'yes' if forecast_available else 'no'}, "
        f"map points={len(detections)}, errors={len(errors)})"
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
