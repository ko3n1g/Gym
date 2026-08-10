# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib import resources
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from referencing import Registry, Resource

from nemo_gym.config_types import ConfigError, Domain
from nemo_gym.environment.manifest import (
    EnvironmentManifest,
    IntegrationProfile,
    ManifestError,
    dump_manifest,
    load_manifest,
    manifest_json_schema,
)


REPO_ROOT = Path(__file__).parents[2]


def _manifest(*, profile: str = "stock-loop", kind: str = "environment") -> dict:
    manifest = {
        "name": "my_eval",
        "version": "1.0.0",
        "kind": kind,
        "integration_profile": profile,
        "domain": "math",
        "description": "Competition mathematics evaluation",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["alice"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "mcqa",
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [
            {
                "name": "validation",
                "type": "validation",
                "jsonl_fpath": "data/validation.jsonl",
            }
        ],
    }
    if profile == "custom-driver":
        manifest["rollout_driver"] = "my_eval.driver:collect"
    if kind == "benchmark":
        manifest["canonical_split"] = "validation"
        manifest["standard_prompt_config"] = "prompts/standard.yaml"
        manifest["datasets"][0]["type"] = "benchmark"
        manifest["datasets"][0]["prepare_script"] = "prepare.py"
        manifest["datasets"][0]["prompt_config"] = "prompts/standard.yaml"
    return manifest


def test_environment_manifest_parses_and_uses_declared_defaults() -> None:
    raw = _manifest()
    raw.pop("licensing")
    raw.pop("determinism")
    manifest = EnvironmentManifest.model_validate(raw)

    assert manifest.name == "my_eval"
    assert manifest.licensing == "unknown"
    assert manifest.determinism.value == "unknown"
    assert manifest.lifecycle.value == "active"
    assert manifest.session_model is None


def test_integration_profiles_are_closed_string_values() -> None:
    assert [profile.value for profile in IntegrationProfile] == [
        "stock-loop",
        "measured-loop",
        "external-loop",
        "custom-driver",
    ]
    assert str(IntegrationProfile.STOCK_LOOP) == "stock-loop"


@pytest.mark.parametrize("missing_field", ["resources_server", "agent_server", "datasets"])
def test_manifest_requires_base_composition(missing_field: str) -> None:
    raw = _manifest()
    raw.pop(missing_field)

    with pytest.raises(ValidationError, match=missing_field):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize("profile", ["stock-loop", "measured-loop"])
def test_in_process_profiles_require_a_model_server(profile: str) -> None:
    raw = _manifest(profile=profile)
    raw.pop("model_server")

    with pytest.raises(ValidationError, match="model_server"):
        EnvironmentManifest.model_validate(raw)


def test_external_loop_may_omit_a_model_server() -> None:
    raw = _manifest(profile="external-loop")
    raw.pop("model_server")

    assert EnvironmentManifest.model_validate(raw).model_server is None


def test_custom_driver_requires_a_rollout_driver() -> None:
    raw = _manifest(profile="custom-driver")
    raw.pop("rollout_driver")

    with pytest.raises(ValidationError, match="rollout_driver"):
        EnvironmentManifest.model_validate(raw)


def test_rollout_driver_is_custom_profile_only() -> None:
    stock = _manifest()
    stock["rollout_driver"] = "package.driver:collect"
    with pytest.raises(ValidationError, match="only valid for the custom-driver"):
        EnvironmentManifest.model_validate(stock)

    custom = _manifest(profile="custom-driver")
    custom["rollout_driver"] = "not a callable"
    with pytest.raises(ValidationError, match="rollout_driver"):
        EnvironmentManifest.model_validate(custom)


@pytest.mark.parametrize("missing_field", ["canonical_split", "standard_prompt_config"])
def test_benchmark_requires_protocol_fields(missing_field: str) -> None:
    raw = _manifest(kind="benchmark")
    raw.pop(missing_field)

    with pytest.raises(ValidationError, match=missing_field):
        EnvironmentManifest.model_validate(raw)


def test_benchmark_requires_a_benchmark_dataset() -> None:
    raw = _manifest(kind="benchmark")
    raw["datasets"][0]["type"] = "validation"

    with pytest.raises(ValidationError, match="benchmark dataset"):
        EnvironmentManifest.model_validate(raw)

    raw = _manifest(kind="benchmark")
    raw["datasets"][0].pop("prepare_script")
    with pytest.raises(ValidationError, match="prepare_script"):
        EnvironmentManifest.model_validate(raw)

    raw = _manifest(kind="benchmark")
    raw["datasets"][0].pop("prompt_config")
    assert EnvironmentManifest.model_validate(raw).datasets[0].prompt_config is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("name", "../escape"),
        ("integration_profile", "invented-loop"),
        ("licensing", "not an SPDX id"),
        ("licensing", "FooBar"),
        ("licensing", "LicenseRef-Internal-Evaluation"),
    ],
)
def test_manifest_rejects_invalid_scalar_contracts(field: str, value: str) -> None:
    raw = _manifest()
    raw[field] = value

    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize("version", ["0.1.0", "1.2.3-alpha.1+build.5", "10.20.30-rc.1"])
def test_manifest_accepts_semantic_versions(version: str) -> None:
    raw = _manifest()
    raw["version"] = version

    assert EnvironmentManifest.model_validate(raw).version == version


@pytest.mark.parametrize("version", ["1.0", "01.2.3", "1.02.3", "1.2.03", "1.2.3-01", "v1.2.3"])
def test_manifest_rejects_non_semantic_versions(version: str) -> None:
    raw = _manifest()
    raw["version"] = version

    with pytest.raises(ValidationError, match="version"):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize("licensing", ["MIT", "Apache-2.0", "internal", "proprietary", "unknown"])
def test_manifest_accepts_spdx_and_private_license_values(licensing: str) -> None:
    raw = _manifest()
    raw["licensing"] = licensing

    assert EnvironmentManifest.model_validate(raw).licensing == licensing


def test_manifest_rejects_bad_reward_duplicate_authors_and_unknown_fields() -> None:
    raw = _manifest()
    raw["reward"]["range"] = [1, 1]
    raw["authors"] = ["alice", "alice"]
    raw["surprise"] = True

    with pytest.raises(ValidationError) as error:
        EnvironmentManifest.model_validate(raw)
    message = str(error.value)
    assert "lower < upper" in message
    assert "authors must be unique" in message
    assert "Extra inputs are not permitted" in message


def test_manifest_rejects_duplicate_dataset_names() -> None:
    raw = _manifest()
    raw["datasets"].append(dict(raw["datasets"][0]))

    with pytest.raises(ValidationError, match="dataset names must be unique"):
        EnvironmentManifest.model_validate(raw)


def test_adopted_from_validates_source_format() -> None:
    raw = _manifest()
    raw["adopted_from"] = {
        "source": "https://github.com/example/project.git",
        "ref": "v1.2.0",
        "reconciled": "2026-08-01",
    }
    assert EnvironmentManifest.model_validate(raw).adopted_from.ref == "v1.2.0"

    raw["adopted_from"]["source"] = "github.com/example/project"
    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(raw)


def test_load_dump_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(dump_manifest(_manifest()))
    assert load_manifest(path).name == "my_eval"

    with pytest.raises(ManifestError, match="was not found"):
        load_manifest(tmp_path / "missing.yaml")

    path.write_text("name: [broken\n")
    with pytest.raises(ManifestError, match="Malformed YAML"):
        load_manifest(path)

    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ManifestError, match="expected a YAML mapping"):
        load_manifest(path)

    path.write_text("version: nope\n")
    with pytest.raises(ConfigError, match=r"name.*Field required"):
        load_manifest(path)

    with pytest.raises(ManifestError, match="<memory>"):
        dump_manifest({"name": "incomplete"})


def test_checked_in_schema_matches_model_and_external_validation_contract() -> None:
    schema = manifest_json_schema()
    schema_directory = REPO_ROOT / "nemo_gym/environment/schemas"
    schema_path = schema_directory / "environment-manifest.schema.json"
    checked_in = json.loads(schema_path.read_text())

    assert checked_in == schema
    assert schema["$defs"]["Domain"]["enum"] == [domain.value for domain in Domain]
    Draft202012Validator.check_schema(schema)

    def retrieve_schema(uri: str) -> Resource:
        path = Path(unquote(urlparse(uri).path))
        return Resource.from_contents(json.loads(path.read_text()))

    registry = Registry(retrieve=retrieve_schema)
    validator = Draft202012Validator(
        schema,
        registry=registry,
        _resolver=registry.resolver(schema_path.as_uri()),
    )
    assert not list(validator.iter_errors(_manifest()))

    invalid = _manifest()
    invalid["licensing"] = "FooBar"
    assert list(validator.iter_errors(invalid))

    invalid_manifests = []

    for profile in ("stock-loop", "measured-loop"):
        invalid = _manifest(profile=profile)
        invalid.pop("model_server")
        invalid_manifests.append(invalid)

    invalid = _manifest(profile="custom-driver")
    invalid.pop("rollout_driver")
    invalid_manifests.append(invalid)

    invalid = _manifest()
    invalid["rollout_driver"] = "package.driver:collect"
    invalid_manifests.append(invalid)

    for field in ("canonical_split", "standard_prompt_config"):
        invalid = _manifest(kind="benchmark")
        invalid.pop(field)
        invalid_manifests.append(invalid)

    invalid = _manifest(kind="benchmark")
    invalid["datasets"][0]["type"] = "validation"
    invalid_manifests.append(invalid)

    assert all(list(validator.iter_errors(invalid)) for invalid in invalid_manifests)


def test_neutral_example_matches_manifest_contract() -> None:
    manifest = load_manifest(REPO_ROOT / "examples/environment_manifest.yaml")

    assert manifest.name == "example_environment"
    assert manifest.integration_profile == IntegrationProfile.STOCK_LOOP


def test_schema_resources_are_packaged_with_environment_module() -> None:
    schema_directory = resources.files("nemo_gym.environment").joinpath("schemas")

    assert schema_directory.joinpath("environment-manifest.schema.json").is_file()
    assert schema_directory.joinpath("spdx-license-identifier.schema.json").is_file()
