"""Offline-safety tests (issues #12/#19; docs/TESTING.md offline quarantine).

Static and behavioral checks that the pipeline never reaches for the network
inside the offline compute path, never writes into the (read-only) source
directory, and fails clearly when optional dependencies or artifacts are
missing instead of hanging or silently degrading.
"""

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

#: Modules that must never be imported by src/ (offline compute rule).
BANNED_NETWORK_MODULES = {
    "requests",
    "urllib",
    "urllib3",
    "http",
    "httpx",
    "socket",
    "ftplib",
    "smtplib",
    "telnetlib",
    "paramiko",
    "boto3",
    "huggingface_hub",
    "datasets",
}


def iter_python_sources(*roots):
    for root in roots:
        yield from sorted(root.rglob("*.py"))


class TestNoNetworkImports:
    def test_src_never_imports_network_modules(self):
        offenders = []
        for path in iter_python_sources(SRC_ROOT):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                for name in names:
                    if name in BANNED_NETWORK_MODULES:
                        offenders.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
        assert not offenders, f"offline compute path must not import network modules: {offenders}"

    def test_torch_is_imported_lazily_in_patchtst(self):
        source = (SRC_ROOT / "models" / "patchtst.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_level_imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_level_imports.append(node.module or "")
        assert not any("torch" in name for name in module_level_imports), (
            "torch must stay a lazy import so calendar baselines work without it"
        )

    def test_import_all_src_modules_with_torch_blocked(self, monkeypatch):
        """Every src module imports cleanly even when torch is absent."""
        import importlib

        monkeypatch.setitem(sys.modules, "torch", None)
        modules = [
            "src",
            "src.common.config",
            "src.data.schema",
            "src.data.adapters",
            "src.data.gate",
            "src.data.aggregation",
            "src.geo.aoi",
            "src.geo.matching",
            "src.harmonization.features",
            "src.harmonization.baselines",
            "src.harmonization.model",
            "src.harmonization.evaluate",
            "src.harmonization.train",
            "src.harmonization.inference",
            "src.calendar.seasonal",
            "src.calendar.anomalies",
            "src.calendar.forecast",
            "src.calendar.evaluation",
            "src.calendar.records",
            "src.models.registry",
            "src.models.patchtst",
            "src.artifacts.contracts",
            "src.artifacts.manifest",
            "src.artifacts.serialization",
        ]
        for name in modules:
            importlib.import_module(name)  # raises if a module-level torch import slipped in

    def test_calendar_baselines_work_without_torch(self, monkeypatch, synthetic_daily):
        monkeypatch.setitem(sys.modules, "torch", None)
        from src.calendar.forecast import SeasonalNaiveForecaster
        from src.common.config import ForecastConfig

        model = SeasonalNaiveForecaster(ForecastConfig())
        model.fit(synthetic_daily)
        result = model.predict()
        assert len(result["point"]) == 14


class TestSourceDirectoryIsNeverWritten:
    def test_gate_leaves_source_directory_untouched(self, sample_dir, harmonization_config):
        before = {p.name: p.stat().st_mtime for p in sorted(sample_dir.iterdir())}
        from src.data.gate import FivePassDataGate

        FivePassDataGate(harmonization_config).run(str(sample_dir))
        after = {p.name: p.stat().st_mtime for p in sorted(sample_dir.iterdir())}
        assert before == after

    def test_pipeline_outputs_never_land_in_source_dir(self, sample_dir, tmp_path, harmonization_config):
        from src.harmonization.train import run_harmonization_training

        source_files_before = set(p.name for p in sample_dir.iterdir())
        run_harmonization_training(
            str(sample_dir), config=harmonization_config, aoi=None
        )
        source_files_after = set(p.name for p in sample_dir.iterdir())
        assert source_files_before == source_files_after


class TestGracefulFailures:
    def test_missing_harmonizer_artifact_clear_error(self, tmp_path):
        from src.artifacts.serialization import ArtifactIntegrityError, load_harmonizer_bundle

        with pytest.raises(ArtifactIntegrityError, match="missing"):
            load_harmonizer_bundle(tmp_path)

    def test_missing_forecaster_artifact_clear_error(self, tmp_path):
        from src.artifacts.serialization import ArtifactIntegrityError, load_forecaster_bundle

        with pytest.raises(ArtifactIntegrityError):
            load_forecaster_bundle(tmp_path)

    def test_missing_torch_error_mentions_baselines_and_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        from src.models.patchtst import PatchTSTForecaster, TorchDependencyError

        with pytest.raises(TorchDependencyError) as excinfo:
            PatchTSTForecaster().fit(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=200), "value": 1.0}))
        message = str(excinfo.value)
        assert "wheelhouse" in message or "pip install" in message
        assert "baselines remain available" in message

    def test_registry_lists_optional_dependencies(self):
        from src.models.registry import MODEL_REGISTRY

        patchtst = MODEL_REGISTRY.describe()["patchtst"]
        assert patchtst["requires"] == "torch"
        assert MODEL_REGISTRY.describe("forecaster")["persistence"]["requires"] is None
