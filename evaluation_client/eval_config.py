from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class EvalConfig:
    evaluator_email: str
    institution: str
    logging_server_ip: str
    third_person_camera: str
    evaluator_code: str = ""
    cameras: dict = field(default_factory=dict)


def load_config(config_file_path: str) -> EvalConfig:
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    return EvalConfig(
        evaluator_email=data["evaluator_email"],
        institution=data["institution"],
        logging_server_ip=data["logging_server_ip"],
        third_person_camera=data["third_person_camera"],
        evaluator_code=(data.get("evaluator_code") or ""),
        cameras={cam["name"]: cam["id"] for cam in data.get("cameras", [])},
    )


def save_evaluator_code(config_file_path: str, evaluator_code: str) -> None:
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if data.get("evaluator_code") == evaluator_code:
        return

    data["evaluator_code"] = evaluator_code
    with open(config_file_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
