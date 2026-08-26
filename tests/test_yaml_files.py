from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

YAML_FILES = [
    PROJECT_ROOT / "config" / "config.yml",
    PROJECT_ROOT / "environment" / "conda.yml",
    PROJECT_ROOT / "azureml" / "endpoint.yml",
    PROJECT_ROOT / "azureml" / "deployment.yml",
    PROJECT_ROOT / ".github" / "workflows" / "ci.yml",
]


def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def test_required_yaml_files_exist():
    for file_path in YAML_FILES:
        assert file_path.exists(), f"Missing YAML file: {file_path}"


def test_yaml_files_have_valid_syntax():
    for file_path in YAML_FILES:
        yaml_content = load_yaml_file(file_path)

        assert yaml_content is not None, (
            f"YAML file is empty: {file_path}"
        )

        assert isinstance(yaml_content, dict), (
            f"YAML root must be a dictionary: {file_path}"
        )


def test_config_contains_required_sections():
    config_path = PROJECT_ROOT / "config" / "config.yml"
    config = load_yaml_file(config_path)

    required_sections = [
        "data",
        "target",
        "features",
        "model",
        "training",
        "winsorization",
        "hyperparameters",
        "outputs",
    ]

    for section in required_sections:
        assert section in config, (
            f"Missing required config section: {section}"
        )


def test_target_is_riesgo_24():
    config_path = PROJECT_ROOT / "config" / "config.yml"
    config = load_yaml_file(config_path)

    assert config["target"]["name"] == "riesgo_24"


def test_model_type_is_logistic_regression():
    config_path = PROJECT_ROOT / "config" / "config.yml"
    config = load_yaml_file(config_path)

    assert config["model"]["model_type"] == "logistic_regression"


def test_model_has_fourteen_features():
    config_path = PROJECT_ROOT / "config" / "config.yml"
    config = load_yaml_file(config_path)

    features = config["features"]["vars_modelo"]

    assert len(features) == 14
    assert len(features) == len(set(features))


def test_winsorization_limits_are_numeric_and_valid():
    config_path = PROJECT_ROOT / "config" / "config.yml"
    config = load_yaml_file(config_path)

    winsor_cuts = config["winsorization"]["cuts"]

    for feature, limits in winsor_cuts.items():
        assert isinstance(limits, list), (
            f"Winsorization limits for {feature} must be a list."
        )

        assert len(limits) == 2, (
            f"{feature} must have lower and upper limits."
        )

        lower_limit, upper_limit = limits

        assert isinstance(lower_limit, (int, float)), (
            f"Lower limit for {feature} must be numeric."
        )

        assert isinstance(upper_limit, (int, float)), (
            f"Upper limit for {feature} must be numeric."
        )

        assert 0 <= lower_limit < upper_limit <= 1, (
            f"Invalid winsorization limits for {feature}: {limits}"
        )