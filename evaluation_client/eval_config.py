from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class EvalConfig:
    evaluator_email: str
    institution: str
    logging_server_ip: str
    third_person_camera: str
    cameras: dict = field(default_factory=dict)


def load_config(config_file_path: str) -> EvalConfig:
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    return EvalConfig(
        evaluator_email=data["evaluator_email"],
        institution=data["institution"],
        logging_server_ip=data["logging_server_ip"],
        third_person_camera=data["third_person_camera"],
        cameras={cam["name"]: cam["id"] for cam in data.get("cameras", [])},
    )
