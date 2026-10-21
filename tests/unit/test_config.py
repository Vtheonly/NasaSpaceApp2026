"""Unit tests for configuration loading, defaults, and reproducibility helpers."""

import numpy as np
import pytest

from src.common.config import (
    CalendarConfig,
    ForecastConfig,
    HarmonizationConfig,
    PatchTSTParams,
    SplitConfig,
    load_yaml_file,
    resolve_code_commit,
    seed_everything,
)


class TestYamlLoading:
    def test_repo_configs_match_dataclass_defaults(self):
        repo_root = resolve_repo_root()
        harmonization = HarmonizationConfig.from_yaml(repo_root / "configs" / "harmonization.yaml")
        assert harmonization == HarmonizationConfig()
        calendar = CalendarConfig.from_yaml(repo_root / "configs" / "calendar.yaml")
        assert calendar == CalendarConfig()

    def test_overrides_apply(self, tmp_path):
        config_file = tmp_path / "c.yaml"
        config_file.write_text(
            "seed: 7\nsplit:\n  train_end: \"2019-12-31\"\n", encoding="utf-8"
        )
        config = HarmonizationConfig.from_yaml(config_file)
        assert config.seed == 7
        assert config.split.train_end == "2019-12-31"
        # Untouched sections keep defaults.
        assert config.split.validation_end == "2023-12-31"

    def test_empty_yaml_yields_defaults(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        assert HarmonizationConfig.from_yaml(empty) == HarmonizationConfig()

    def test_non_mapping_yaml_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml_file(bad)


class TestDefaults:
    def test_split_defaults_follow_testing_doc(self):
        # docs/TESTING.md pass 3: train 2000-2021, validation 2022-2023, test 2024+.
        assert SplitConfig().train_end == "2021-12-31"
        assert SplitConfig().validation_end == "2023-12-31"

    def test_patchtst_defaults_follow_architecture_doc(self):
        # docs/ARCHITECTURE.md Pillar 2: 16-day patches, stride 8, dim 128,
        # four encoder blocks, eight heads, dropout 0.1.
        params = PatchTSTParams()
        assert params.patch_len == 16
        assert params.stride == 8
        assert params.d_model == 128
        assert params.n_encoder_blocks == 4
        assert params.n_heads == 8
        assert params.dropout == 0.1

    def test_horizon_is_14_days(self):
        assert ForecastConfig().horizon_days == 14

    def test_to_dict_from_dict_roundtrip(self):
        for config in (
            HarmonizationConfig(seed=3),
            CalendarConfig(seed=4),
            ForecastConfig(),
        ):
            assert type(config).from_dict(config.to_dict()) == config

    def test_unknown_keys_are_ignored(self):
        config = HarmonizationConfig.from_dict({"mystery_option": 1, "seed": 9})
        assert config.seed == 9


class TestValidation:
    def test_quantiles_need_median(self):
        with pytest.raises(ValueError, match="median"):
            ForecastConfig.from_dict({"quantiles": [0.1, 0.9]})

    def test_quantiles_in_range(self):
        with pytest.raises(ValueError, match="quantiles"):
            ForecastConfig.from_dict({"quantiles": [0.5, 1.2]})

    def test_patch_len_within_context(self):
        with pytest.raises(ValueError, match="patch_len"):
            _ = PatchTSTParams(context_len=20, patch_len=32).n_patches


class TestReproducibility:
    def test_seed_everything_determines_sampling(self):
        seed_everything(123)
        a = np.random.normal(size=5)
        seed_everything(123)
        b = np.random.normal(size=5)
        assert np.array_equal(a, b)

    def test_resolve_code_commit_returns_string(self):
        value = resolve_code_commit()
        assert isinstance(value, str) and len(value) >= 6

    def test_resolve_code_commit_failing_dir_returns_unknown(self, tmp_path):
        assert resolve_code_commit(tmp_path) in {"unknown", resolve_code_commit()}


def resolve_repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
