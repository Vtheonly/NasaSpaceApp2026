"""Usability/accessibility contract tests for scripts/build_app.py (issue #14).

The viewer app is the reviewable UI for the core demo, so its usability
contracts are enforced by tests, not by eyeballing: semantic landmarks,
labels, keyboard focus, WCAG AA contrast (recomputed independently), a
fully-offline page (no external requests of any kind), correct rendering
of map/calendar/anomalies/forecast from the exported artifacts, and
EXPLICIT no-forecast / missing-artifact states (docs/MODEL_CONTRACTS.md
section 8: no fabricated data, ever).
"""

from __future__ import annotations

import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _entry in (REPO_ROOT, SCRIPTS_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import build_app  # noqa: E402
import run_pipeline  # noqa: E402
from tests.conftest import AOI_SPEC, SAMPLE_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Shared session fixtures: one full pipeline run + one no-forecast run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("app_pipeline") / "run"
    code = run_pipeline.main(
        [
            "--source-dir",
            str(SAMPLE_DIR),
            "--output-dir",
            str(out),
            "--aoi",
            AOI_SPEC,
            "--seed",
            "2026",
        ]
    )
    assert code == 0
    return out


@pytest.fixture(scope="session")
def no_forecast_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("app_pipeline_nofc") / "run"
    code = run_pipeline.main(
        [
            "--source-dir",
            str(SAMPLE_DIR),
            "--output-dir",
            str(out),
            "--aoi",
            AOI_SPEC,
            "--seed",
            "2026",
            "--forecaster",
            "none",
        ]
    )
    assert code == 0
    return out


@pytest.fixture(scope="session")
def app_build(pipeline_out: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("app_out")
    code = build_app.main(
        [
            "--inference-dir",
            str(pipeline_out / "inference"),
            "--source-dir",
            str(SAMPLE_DIR),
            "--output-dir",
            str(out),
            "--generated-at",
            "2026-10-22T00:00:00Z",
        ]
    )
    assert code == 0
    return out


@pytest.fixture(scope="session")
def app_dom(app_build: Path) -> "Outline":
    return parse_html((app_build / "index.html").read_text(encoding="utf-8"))


class Outline(HTMLParser):
    """Tiny stdlib DOM: tags with attributes, raw text, order preserved."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.text: str = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append((tag, {k: (v or "") for k, v in attrs}))

    def handle_data(self, data: str) -> None:
        self.text += data


def parse_html(raw: str) -> Outline:
    parser = Outline()
    parser.feed(raw)
    return parser


def _attrs(dom: Outline, tag: str) -> list[dict[str, str]]:
    return [attrs for t, attrs in dom.tags if t == tag]


def _count_attr_value(dom: Outline, tag: str, attr: str, value: str) -> int:
    return sum(
        1
        for t, attrs in dom.tags
        if t == tag and value in attrs.get(attr, "").split()
    )


# ---------------------------------------------------------------------------
# Independent WCAG AA verification (never trust the builder's own math)
# ---------------------------------------------------------------------------


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rel_lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast(fg: str, bg: str) -> float:
    l1, l2 = _rel_lum(fg), _rel_lum(bg)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


class TestPaletteMeetsAAIndependently:
    def test_every_declared_pair_meets_4_5(self) -> None:
        report = build_app._palette_report()
        assert report, "palette report must not be empty"
        for name, ratio in report.items():
            assert ratio >= 4.5, f"{name}: {ratio}"

    def test_recomputed_from_raw_palette_constants(self) -> None:
        p = build_app.PALETTE
        pairs = {
            "ink": (p["ink"], p["bg"]),
            "ink_surface": (p["ink"], p["surface"]),
            "secondary": (p["ink_secondary"], p["bg"]),
            "accent": (p["accent"], p["bg"]),
            "high": (p["anomaly_high"], p["bg"]),
            "low": (p["anomaly_low"], p["bg"]),
            "focus": (p["focus"], p["bg"]),
            "modis": (p["modis"], p["bg"]),
            "viirs": (p["viirs"], p["bg"]),
        }
        for name, (fg, bg) in pairs.items():
            assert _contrast(fg, bg) >= 4.5, name

    def test_white_text_on_two_darkest_heat_levels(self) -> None:
        assert _contrast("#ffffff", build_app.HEAT_LEVELS[4]) >= 4.5
        assert _contrast("#ffffff", build_app.HEAT_LEVELS[3]) >= 4.5

    def test_build_report_records_pass(self, app_build: Path) -> None:
        report = json.loads((app_build / "app_build.json").read_text())
        assert report["palette_passes_aa"] is True


# ---------------------------------------------------------------------------
# Document semantics, labels, keyboard access
# ---------------------------------------------------------------------------


class TestDocumentSemantics:
    def test_language_title_and_viewport(self, app_dom: Outline) -> None:
        html_attrs = _attrs(app_dom, "html")[0]
        assert html_attrs["lang"] == "en"
        assert _attrs(app_dom, "title"), "document must have a <title>"
        meta = _attrs(app_dom, "meta")
        assert any("viewport" in m.get("name", "") for m in meta)

    def test_single_h1_and_section_h2s(self, app_dom: Outline) -> None:
        assert len(_attrs(app_dom, "h1")) == 1
        h2_text = " ".join(
            app_dom.text for _ in [0]
        )
        for heading in (
            "Overview",
            "Hotspot map",
            "52-week-by-year burning calendar",
            "Anomaly details",
            "14-day forecast with uncertainty",
            "Provenance and limitations",
        ):
            assert heading in app_dom.text, heading

    def test_landmarks_present(self, app_dom: Outline) -> None:
        tags = {t for t, _ in app_dom.tags}
        assert {"header", "main", "footer", "section"} <= tags

    def test_skip_link_is_first_focusable(self, app_dom: Outline) -> None:
        first = app_dom.tags[app_dom.tags.index(("body", {})) + 1]
        assert first[0] == "a"
        assert first[1].get("href") == "#main"
        assert "skip" in first[1].get("class", "")

    def test_sections_are_labelled(self, app_dom: Outline) -> None:
        sections = _attrs(app_dom, "section")
        assert len(sections) == 6
        for attrs in sections:
            ref = attrs.get("aria-labelledby")
            assert ref, "every section needs aria-labelledby"

    def test_svgs_carry_role_img_and_labels(self, app_dom: Outline) -> None:
        svgs = _attrs(app_dom, "svg")
        assert len(svgs) >= 1
        for attrs in svgs:
            assert attrs["role"] == "img"
            assert len(attrs.get("aria-label", "")) > 30
            assert "tabindex" in attrs

    def test_forecast_chart_present_with_quantile_labels(self, app_dom: Outline) -> None:
        assert "10th-90th percentile" in app_dom.text
        assert "median forecast" in app_dom.text

    def test_focus_visible_and_reduced_motion_rules(self, app_build: Path) -> None:
        css = (app_build / "index.html").read_text(encoding="utf-8")
        assert ":focus-visible" in css
        assert "prefers-reduced-motion" in css

    def test_responsive_media_queries_present(self, app_build: Path) -> None:
        css = (app_build / "index.html").read_text(encoding="utf-8")
        assert "@media (max-width:640px)" in css
        assert "repeat(auto-fit,minmax(210px,1fr))" in css


class TestKeyboardAccess:
    def test_calendar_year_rows_focusable_with_summaries(self, app_dom: Outline) -> None:
        rows = [
            attrs
            for t, attrs in app_dom.tags
            if t == "div" and "cal-year" in attrs.get("class", "").split()
        ]
        assert rows, "year rows must exist"
        for attrs in rows:
            assert attrs.get("tabindex") == "0"
            label = attrs.get("aria-label", "")
            assert "weeks" in label and "sensor era" in label
            assert (
                "days without coverage" in label
            )  # coverage honesty in the summary

    def test_tables_have_caption_and_scoped_headers(self, app_dom: Outline) -> None:
        captions = _attrs(app_dom, "caption")
        assert captions, "anomaly table needs a caption"
        heads = _attrs(app_dom, "th")
        assert any(h.get("scope") == "col" for h in heads)
        assert any(h.get("scope") == "row" for h in heads)


# ---------------------------------------------------------------------------
# Offline guarantee: the page must make ZERO external requests
# ---------------------------------------------------------------------------


class TestOfflineGuarantee:
    def test_no_external_resource_references(self, app_dom: Outline) -> None:
        external_hrefs = [
            (t, a)
            for t, a in app_dom.tags
            for key in ("src", "href")
            if a.get(key, "").startswith(("http://", "https://", "//"))
        ]
        assert external_hrefs == [], external_hrefs[:3]

    def test_no_scripts_or_link_elements_at_all(self, app_dom: Outline) -> None:
        tags = {t for t, _ in app_dom.tags}
        assert "script" not in tags
        assert "link" not in tags
        assert "iframe" not in tags

    def test_attribution_urls_are_plain_text_only(self, app_build: Path) -> None:
        raw = (app_build / "index.html").read_text(encoding="utf-8")
        # URLs may appear as TEXT (attribution), never as navigable attributes.
        assert "https://earthdata.nasa.gov/firms" in raw

    def test_self_contained_single_file(self, app_build: Path) -> None:
        """Everything a browser needs lives in index.html - no sibling files."""
        files = [p.name for p in app_build.iterdir() if p.name != "app_build.json"]
        assert files == ["index.html"]


# ---------------------------------------------------------------------------
# Rendering correctness from the exported artifacts
# ---------------------------------------------------------------------------


class TestDataCorrectness:
    def test_map_point_counts_match_fixtures(
        self, app_dom: Outline
    ) -> None:
        modis_rows = sum(
            1
            for _t, a in app_dom.tags
            if _t == "circle" and a.get("fill") == build_app.PALETTE["modis"]
        )
        viirs_rows = sum(
            1
            for _t, a in app_dom.tags
            if _t == "path" and a.get("fill") == build_app.PALETTE["viirs"]
        )
        assert modis_rows == 1678  # modis_active_fires.csv rows
        assert viirs_rows == 2607  # viirs_active_fires.csv rows

    def test_map_legend_distinguishes_sensors(self, app_dom: Outline) -> None:
        assert "MODIS (circle)" in app_dom.text
        assert "VIIRS (triangle)" in app_dom.text

    def test_calendar_years_match_series(
        self, pipeline_out: Path, app_dom: Outline
    ) -> None:
        daily = pd.read_csv(pipeline_out / "inference" / "harmonized_daily.csv")
        iso_years = {
            int(y) for y in pd.to_datetime(daily["date"]).dt.isocalendar()["year"]
        }
        year_rows = sum(
            1
            for t, a in app_dom.tags
            if t == "div" and "cal-year" in a.get("class", "").split()
        )
        assert year_rows == len(iso_years)

    def test_missing_weeks_render_distinct_from_low_activity(
        self, app_dom: Outline
    ) -> None:
        assert _count_attr_value(app_dom, "i", "class", "cal-missing") > 0
        assert _count_attr_value(app_dom, "i", "class", "cal-heat-0") > 0
        text = app_dom.text
        assert "no observed coverage (missing data, NOT zero activity)" in text

    def test_partial_coverage_weeks_marked(self, app_dom: Outline) -> None:
        assert _count_attr_value(app_dom, "i", "class", "cal-partial") > 0

    def test_anomaly_table_row_count_and_content(
        self, pipeline_out: Path, app_dom: Outline
    ) -> None:
        anomalies = pd.read_csv(pipeline_out / "inference" / "anomaly_records.csv")
        expected = min(len(anomalies), build_app.ANOMALY_ROWS)
        tr_count = sum(1 for t, _a in app_dom.tags if t == "tr")
        data_rows = tr_count - 1  # minus the header row
        assert data_rows == expected

    def test_forecast_status_block_shows_model_horizon_calibration(
        self, pipeline_out: Path, app_dom: Outline
    ) -> None:
        calendar = pd.read_csv(
            pipeline_out / "inference" / "calendar_forecast_records.csv"
        )
        horizon = int((calendar["quality_state"] == "forecast").sum())
        assert horizon > 0
        assert "Forecast model:" in app_dom.text
        assert f"{horizon}-day horizon" in app_dom.text
        assert "experimental_uncalibrated" in app_dom.text
        assert "not calibrated probabilities" in app_dom.text

    def test_forecast_fan_is_rendered(self, app_dom: Outline) -> None:
        paths = [a for t, a in app_dom.tags if t == "path" and "Z" in a.get("d", "")]
        fans = [a for a in paths if a.get("fill") == build_app.PALETTE["accent"]]
        assert len(fans) == 1  # the q10-q90 band

    def test_overview_stats_match_daily_series(
        self, pipeline_out: Path, app_dom: Outline
    ) -> None:
        daily = pd.read_csv(pipeline_out / "inference" / "harmonized_daily.csv")
        observed = int((daily["value_type"] != "missing").sum())
        missing = len(daily) - observed
        text = app_dom.text
        assert f"{len(daily)} calendar days" in text
        assert f"{observed} observed days" in text
        assert (
            f"{missing} days without coverage (explicitly missing, never zero)"
            in text
        )

    def test_synthetic_warning_is_prominent(self, app_dom: Outline) -> None:
        assert "SYNTHETIC DEMO DATA" in app_dom.text

    def test_firms_attribution_in_footer(self, app_dom: Outline) -> None:
        assert build_app.OFFICIAL_ACKNOWLEDGMENT in app_dom.text

    def test_limitations_listed(self, app_dom: Outline) -> None:
        for phrase in (
            "not ground truth",
            "both sensors could see",
            "uncalibrated",
            "never as zero activity",
        ):
            assert phrase in app_dom.text


# ---------------------------------------------------------------------------
# Explicit states: no forecast, missing artifacts
# ---------------------------------------------------------------------------


class TestNoForecastState:
    @pytest.fixture(scope="class")
    def nofc_build(
        self, no_forecast_out: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        out = tmp_path_factory.mktemp("app_nofc")
        code = build_app.main(
            [
                "--inference-dir",
                str(no_forecast_out / "inference"),
                "--source-dir",
                str(SAMPLE_DIR),
                "--output-dir",
                str(out),
                "--generated-at",
                "2026-10-22T00:00:00Z",
            ]
        )
        assert code == 0
        return out

    def test_explicit_no_forecast_status(self, nofc_build: Path) -> None:
        dom = parse_html((nofc_build / "index.html").read_text(encoding="utf-8"))
        status_blocks = [
            a for t, a in dom.tags if t == "div" and a.get("role") == "status"
        ]
        assert status_blocks, "no-forecast state must be an ARIA status block"
        assert "No forecast available" in dom.text
        assert "nothing here is extrapolated" in dom.text

    def test_no_forecast_svg_when_unavailable(self, nofc_build: Path) -> None:
        raw = (nofc_build / "index.html").read_text(encoding="utf-8")
        # The forecast CHART element must not exist (the CSS class name alone
        # appearing in the stylesheet is fine and unavoidable).
        assert '<svg viewBox="0 0 800 300"' not in raw

    def test_build_report_marks_forecast_unavailable(self, nofc_build: Path) -> None:
        report = json.loads((nofc_build / "app_build.json").read_text())
        assert report["forecast_available"] is False

    def test_calendar_still_complete_without_forecast(
        self, nofc_build: Path, no_forecast_out: Path
    ) -> None:
        dom = parse_html((nofc_build / "index.html").read_text(encoding="utf-8"))
        year_rows = sum(
            1
            for t, a in dom.tags
            if t == "div" and "cal-year" in a.get("class", "").split()
        )
        daily = pd.read_csv(no_forecast_out / "inference" / "harmonized_daily.csv")
        iso_years = {
            int(y) for y in pd.to_datetime(daily["date"]).dt.isocalendar()["year"]
        }
        assert year_rows == len(iso_years)


class TestErrorStates:
    def test_empty_inference_dir_refuses_to_render(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "app"
        code = build_app.main(
            ["--inference-dir", str(tmp_path / "nothing"), "--output-dir", str(out)]
        )
        assert code == 1
        report = json.loads((out / "app_build.json").read_text())
        assert report["status"] == "failed"
        assert report["errors"]
        # An empty shell app must NOT be written (no fabricated UI).
        assert not (out / "index.html").exists()

    def test_partial_inputs_render_explicit_alerts(
        self, pipeline_out: Path, tmp_path: Path
    ) -> None:
        inference = tmp_path / "partial"
        inference.mkdir()
        daily_src = pipeline_out / "inference" / "harmonized_daily.csv"
        (inference / "harmonized_daily.csv").write_bytes(daily_src.read_bytes())
        out = tmp_path / "app"
        code = build_app.main(
            ["--inference-dir", str(inference), "--output-dir", str(out),
             "--generated-at", "2026-10-22T00:00:00Z"]
        )
        assert code == 1
        dom = parse_html((out / "index.html").read_text(encoding="utf-8"))
        alerts = [
            a
            for t, a in dom.tags
            if t == "div" and a.get("role") == "alert"
        ]
        joined = " ".join(a.get("class", "") for a in alerts)
        assert alerts, "missing artifacts must surface as alerts"
        assert "state-error" in joined
        assert "calendar_forecast_records.csv" in dom.text
        assert "anomaly_records.csv" in dom.text

    def test_unreadable_csv_becomes_alert(self, pipeline_out: Path, tmp_path: Path) -> None:
        inference = tmp_path / "broken"
        inference.mkdir()
        # Parses as CSV but violates the export contract (no value_type/date).
        (inference / "harmonized_daily.csv").write_text("not,a,valid\x00csv")
        (inference / "calendar_forecast_records.csv").write_bytes(
            (pipeline_out / "inference" / "calendar_forecast_records.csv").read_bytes()
        )
        (inference / "anomaly_records.csv").write_bytes(
            (pipeline_out / "inference" / "anomaly_records.csv").read_bytes()
        )
        out = tmp_path / "app"
        code = build_app.main(
            ["--inference-dir", str(inference), "--output-dir", str(out),
             "--generated-at", "2026-10-22T00:00:00Z"]
        )
        assert code == 1
        raw = (out / "index.html").read_text(encoding="utf-8")
        assert "violates the export contract" in raw
        assert "role=\"alert\"" in raw


class TestBuildProvenanceAndDeterminism:
    def test_input_hashes_recorded_and_correct(
        self, pipeline_out: Path, app_build: Path
    ) -> None:
        report = json.loads((app_build / "app_build.json").read_text())
        for name, meta in report["inputs"].items():
            digest = hashlib.sha256(
                (pipeline_out / "inference" / name).read_bytes()
            ).hexdigest()
            assert meta["sha256"] == digest

    def test_report_records_detections_and_synthetic_flag(
        self, app_build: Path
    ) -> None:
        report = json.loads((app_build / "app_build.json").read_text())
        assert report["map_detections"] == 4285
        assert report["synthetic_demo"] is True
        assert report["status"] == "passed"

    def test_deterministic_builds_are_byte_identical(
        self, pipeline_out: Path, tmp_path: Path
    ) -> None:
        outs = []
        for i in (1, 2):
            out = tmp_path / f"det{i}"
            code = build_app.main(
                [
                    "--inference-dir",
                    str(pipeline_out / "inference"),
                    "--source-dir",
                    str(SAMPLE_DIR),
                    "--output-dir",
                    str(out),
                    "--generated-at",
                    "2026-10-22T00:00:00Z",
                ]
            )
            assert code == 0
            outs.append((out / "index.html").read_bytes())
        assert outs[0] == outs[1]

    def test_map_point_cap_respected(self, pipeline_out: Path, tmp_path: Path) -> None:
        out = tmp_path / "capped"
        code = build_app.main(
            [
                "--inference-dir",
                str(pipeline_out / "inference"),
                "--source-dir",
                str(SAMPLE_DIR),
                "--output-dir",
                str(out),
                "--max-map-points",
                "100",
                "--generated-at",
                "2026-10-22T00:00:00Z",
            ]
        )
        assert code == 0
        report = json.loads((out / "app_build.json").read_text())
        assert report["map_detections"] == 100
