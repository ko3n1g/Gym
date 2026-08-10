# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.environment.manifest import EnvironmentManifest, dump_manifest, load_manifest
from nemo_gym.environment.validation import EnvironmentValidationError, validate_environment


def _manifest(*, kind: str = "environment", profile: str = "stock-loop") -> dict:
    root = f"{'benchmarks' if kind == 'benchmark' else 'environments'}/demo"
    dataset = {
        "name": "example",
        "type": "benchmark" if kind == "benchmark" else "example",
        "jsonl_fpath": f"{root}/data/example.jsonl",
        "num_repeats": 1,
    }
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "kind": kind,
        "integration_profile": profile,
        "domain": "math",
        "description": "A small exact-match evaluation.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["contributor"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "demo",
        "agent_server": "simple_agent",
        "datasets": [dataset],
        "grading_mode": "exact",
    }
    if profile in {"stock-loop", "measured-loop"}:
        manifest["model_server"] = "policy_model"
    if kind == "benchmark":
        dataset["prepare_script"] = f"{root}/prepare.py"
        manifest.update(canonical_split="test", standard_prompt_config=f"{root}/prompts/default.yaml")
    if profile == "custom-driver":
        manifest["rollout_driver"] = "environments.demo.rollout_driver:run_rollout_collection"
    return manifest


def _config(*, kind: str = "environment", profile: str = "stock-loop") -> str:
    root = f"{'benchmarks' if kind == 'benchmark' else 'environments'}/demo"
    dataset = f"""      - name: example
        type: {"benchmark" if kind == "benchmark" else "example"}
        jsonl_fpath: {root}/data/example.jsonl
"""
    if kind == "benchmark":
        dataset += f"        prepare_script: {root}/prepare.py\n"
    model_server = (
        ""
        if profile == "custom-driver"
        else """      model_server:
        type: responses_api_models
        name: policy_model
"""
    )
    rollout_driver = (
        "rollout_collection_driver: environments.demo.rollout_driver:run_rollout_collection\n"
        if profile == "custom-driver"
        else ""
    )
    return f"""demo_resources:
  resources_servers:
    demo:
      entrypoint: app.py
      domain: math
      grading_mode: exact
demo_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: demo_resources
{model_server.rstrip()}
      datasets:
{dataset}{rollout_driver}"""


def _asset(tmp_path: Path, *, kind: str = "environment", profile: str = "stock-loop") -> Path:
    directory = tmp_path / ("benchmarks" if kind == "benchmark" else "environments") / "demo"
    (directory / "data").mkdir(parents=True)
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(dump_manifest(_manifest(kind=kind, profile=profile)), encoding="utf-8")
    (directory / "config.yaml").write_text(_config(kind=kind, profile=profile), encoding="utf-8")

    if kind == "benchmark":
        row = {"question": "What is 1 + 1?", "expected_answer": "2"}
        (directory / "prompts").mkdir()
        (directory / "prompts/default.yaml").write_text("user: '{question}'\n", encoding="utf-8")
        (directory / "prepare.py").write_text(
            "from pathlib import Path\n\ndef prepare(output: Path) -> Path:\n    return output\n",
            encoding="utf-8",
        )
    else:
        row = {"responses_create_params": {"input": "What is 1 + 1?"}, "expected_answer": "2"}
    (directory / "data/example.jsonl").write_text(f"{json.dumps(row)}\n", encoding="utf-8")

    if profile == "custom-driver":
        (directory / "rollout_driver.py").write_text(
            "def run_rollout_collection():\n    pass\n",
            encoding="utf-8",
        )
    return manifest_path


def _replace_manifest(path: Path, **changes: object) -> EnvironmentManifest:
    manifest = load_manifest(path)
    updated = manifest.model_copy(update=changes)
    path.write_text(dump_manifest(updated), encoding="utf-8")
    return manifest


def test_reports_resolved_composition_and_declared_profile(tmp_path: Path) -> None:
    report = validate_environment(_asset(tmp_path))

    assert report.name == "demo"
    assert report.declared_profile == "stock-loop"
    assert "integration_profile" not in report.to_dict()
    assert report.datasets[0].rows == 1
    assert report.grading_mode == "exact"
    assert report.rollout_driver is None
    assert [(item.role, item.implementation) for item in report.components] == [
        ("resources_server", "demo"),
        ("agent_server", "simple_agent"),
        ("model_server", "runtime-selected"),
    ]


def test_benchmark_uses_root_prompt_without_executing_prepare(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    prepare_path = manifest_path.parent / "prepare.py"
    prepare_path.write_text(
        "from pathlib import Path\n\ndef prepare(source: Path) -> Path:\n    raise RuntimeError('must not execute')\n",
        encoding="utf-8",
    )

    report = validate_environment(manifest_path)

    assert report.datasets[0].type == "benchmark"
    assert report.datasets[0].prompt_config.endswith("prompts/default.yaml")


def test_malformed_benchmark_prompt_is_an_actionable_validation_error(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prompts/default.yaml").write_text("user: [broken\n", encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match="Could not materialize benchmark dataset"):
        validate_environment(manifest_path)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("async def prepare() -> Path:\n    pass\n", "synchronous prepare"),
        ("def prepare(\n", "Could not parse dataset prepare script"),
    ],
)
def test_prepare_contract_is_checked_statically(source: str, message: str, tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prepare.py").write_text(source, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_prepare_annotations_are_not_a_runtime_requirement(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prepare.py").write_text("def prepare():\n    return 'output'\n", encoding="utf-8")

    validate_environment(manifest_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "other"}, "identity"),
        ({"kind": "benchmark", "canonical_split": "test", "standard_prompt_config": "prompt.yaml"}, "kind"),
    ],
)
def test_manifest_identity_matches_its_catalog_path(changes: dict, message: str, tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    raw = load_manifest(manifest_path).model_dump(mode="json", exclude_none=True)
    raw.update(changes)
    if changes.get("kind") == "benchmark":
        raw["datasets"][0].update(type="benchmark", prepare_script="prepare.py")
    manifest_path.write_text(dump_manifest(raw), encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_slash_qualified_name_matches_nested_catalog_path(tmp_path: Path) -> None:
    config_manifest_path = _asset(tmp_path)
    nested_manifest_path = tmp_path / "environments/acme/demo/manifest.yaml"
    nested_manifest_path.parent.mkdir(parents=True)
    manifest = load_manifest(config_manifest_path).model_copy(update={"name": "acme/demo"})
    nested_manifest_path.write_text(dump_manifest(manifest), encoding="utf-8")

    report = validate_environment(nested_manifest_path, config_manifest_path.with_name("config.yaml"))

    assert report.name == "acme/demo"


def test_config_defaults_to_sibling_and_must_exist(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    validate_environment(manifest_path)
    config_path.unlink()

    with pytest.raises(EnvironmentValidationError, match="config.yaml"):
        validate_environment(manifest_path)


def test_stale_mirrors_are_reported_and_sync_changes_only_them(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    original = load_manifest(manifest_path)
    stale_dataset = original.datasets[0].model_copy(update={"jsonl_fpath": "wrong.jsonl"})
    _replace_manifest(manifest_path, resources_server="wrong", datasets=[stale_dataset])

    with pytest.raises(EnvironmentValidationError, match="resources_server") as caught:
        validate_environment(manifest_path)
    assert '"jsonl_fpath": "wrong.jsonl"' in str(caught.value)

    report = validate_environment(manifest_path, sync=True)
    synchronized = load_manifest(manifest_path)
    assert report.synchronized_fields == ("resources_server", "datasets")
    assert synchronized.resources_server == "demo"
    assert synchronized.description == original.description


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "is empty"),
        ("[broken\n", "Malformed JSON"),
        ("[]\n", "must contain a JSON object"),
        ('{"responses_create_params": {"input": 42}}\n', "responses_create_params.input"),
    ],
)
def test_dataset_row_errors_are_actionable(tmp_path: Path, content: str, message: str) -> None:
    manifest_path = _asset(tmp_path)
    manifest_path.parent.joinpath("data/example.jsonl").write_text(content, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_invalid_dataset_encoding_is_an_actionable_error(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    manifest_path.parent.joinpath("data/example.jsonl").write_bytes(b"\xff\n")

    with pytest.raises(EnvironmentValidationError, match="Could not read dataset"):
        validate_environment(manifest_path)


def test_custom_driver_is_checked_without_scaffolding(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, profile="custom-driver")
    report = validate_environment(manifest_path)
    assert report.rollout_driver == "environments.demo.rollout_driver:run_rollout_collection"

    driver_path = manifest_path.parent / "rollout_driver.py"
    driver_path.write_text("def other_function():\n    pass\n", encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="Rollout driver.*was not found"):
        validate_environment(manifest_path)

    driver_path.unlink()
    with pytest.raises(EnvironmentValidationError, match="Rollout driver module was not found"):
        validate_environment(manifest_path)
