#!/usr/bin/env python3
"""Robot-free RoboArena policy-server inference check.

This script connects directly to one policy server, builds the same kind of
request payload that the real evaluator sends, and verifies that the server
returns a usable action chunk. It does not contact the RoboArena central server
and does not log or count an evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from urllib.parse import urlparse
from typing import Any

import numpy as np
from PIL import Image

import image_tools
from websocket_client_policy import WebsocketClientPolicy


DEFAULT_RAW_IMAGE_HEIGHT = 288
DEFAULT_RAW_IMAGE_WIDTH = 512


def _normalize_host(host: str) -> str:
    host = host.strip()
    parsed = urlparse(host)
    if parsed.scheme and parsed.hostname:
        return parsed.hostname
    if host.startswith("//"):
        parsed = urlparse(f"ws:{host}")
        if parsed.hostname:
            return parsed.hostname
    return host.strip("/")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _metadata_image_resolution(server_cfg: dict[str, Any]) -> tuple[int, int] | None:
    image_resolution = server_cfg.get("image_resolution")
    if image_resolution is None:
        return None
    if len(image_resolution) != 2:
        raise ValueError(
            f"Server metadata image_resolution must have length 2, got {image_resolution!r}"
        )
    return int(image_resolution[0]), int(image_resolution[1])


def _dummy_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[..., 0] = x
    img[..., 1] = y
    img[..., 2] = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return img


def _prepare_image(
    img: np.ndarray,
    image_resolution: tuple[int, int] | None,
) -> np.ndarray:
    if image_resolution is None:
        return image_tools.convert_to_uint8(img)
    height, width = image_resolution
    return image_tools.convert_to_uint8(
        image_tools.resize(img, height, width, method=Image.LANCZOS)
    )


def _build_dummy_request(
    server_cfg: dict[str, Any],
    *,
    prompt: str,
    session_id: str,
    raw_image_height: int,
    raw_image_width: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_resolution = _metadata_image_resolution(server_cfg)
    needs_wrist = bool(server_cfg.get("needs_wrist_camera", True))
    n_external = int(server_cfg.get("n_external_cameras", 1))
    needs_stereo = bool(server_cfg.get("needs_stereo_camera", False))
    include_session_id = bool(server_cfg.get("needs_session_id", False))

    if n_external not in {0, 1, 2}:
        raise ValueError(
            "Server metadata n_external_cameras must be 0, 1, or 2; "
            f"got {n_external!r}"
        )

    request_data: dict[str, Any] = {
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.array([1.0], dtype=np.float32),
        "prompt": prompt,
    }

    def add_image(key: str, offset: int) -> None:
        raw = _dummy_image(raw_image_height, raw_image_width, seed + offset)
        request_data[key] = _prepare_image(raw, image_resolution)

    if n_external >= 1:
        add_image("observation/exterior_image_1_left", 10)
        if needs_stereo:
            add_image("observation/exterior_image_1_right", 11)
    if n_external >= 2:
        add_image("observation/exterior_image_2_left", 12)
        if needs_stereo:
            add_image("observation/exterior_image_2_right", 13)
    if needs_wrist:
        add_image("observation/wrist_image_left", 20)
        if needs_stereo:
            add_image("observation/wrist_image_right", 21)
    if include_session_id:
        request_data["session_id"] = session_id

    summary = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in request_data.items()
        if isinstance(value, np.ndarray)
    }
    summary["prompt"] = prompt
    if include_session_id:
        summary["session_id"] = session_id
    return request_data, summary


def _validate_actions(
    result: dict[str, Any],
    *,
    allow_unexpected_action_dim: bool,
) -> tuple[np.ndarray, list[str]]:
    if "actions" not in result:
        raise ValueError(
            "Inference response did not contain an 'actions' field. "
            f"Response keys: {sorted(result.keys())}"
        )

    actions = np.asarray(result["actions"])
    if actions.ndim == 1:
        actions = actions[None, ...]
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be rank 1 or 2, got shape {actions.shape}")
    if actions.shape[0] < 1:
        raise ValueError("Expected at least one action in the returned action chunk")
    if not np.issubdtype(actions.dtype, np.number):
        raise ValueError(f"Actions must be numeric, got dtype {actions.dtype}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Actions contain NaN or infinite values")

    warnings: list[str] = []
    if actions.shape[-1] not in {7, 8}:
        message = (
            "Expected final action dimension 7 or 8 for RoboArena/DROID-style "
            f"actions, got {actions.shape[-1]}"
        )
        if allow_unexpected_action_dim:
            warnings.append(message)
        else:
            raise ValueError(message)
    return actions, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to one RoboArena policy server and run a robot-free dummy "
            "inference check."
        )
    )
    parser.add_argument("--host", required=True, help="Policy server host or domain")
    parser.add_argument("--port", required=True, type=int, help="Policy server port")
    parser.add_argument(
        "--prompt",
        default="pick up the object",
        help="Natural-language instruction to include in the dummy request",
    )
    parser.add_argument(
        "--num-calls",
        type=int,
        default=1,
        help="Number of inference calls to make on the same websocket connection",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Websocket connection timeout in seconds",
    )
    parser.add_argument(
        "--raw-image-height",
        type=int,
        default=DEFAULT_RAW_IMAGE_HEIGHT,
        help="Height of synthetic raw camera images before metadata resize",
    )
    parser.add_argument(
        "--raw-image-width",
        type=int,
        default=DEFAULT_RAW_IMAGE_WIDTH,
        help="Width of synthetic raw camera images before metadata resize",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for dummy image noise")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id to send if the server declares needs_session_id",
    )
    parser.add_argument(
        "--allow-unexpected-action-dim",
        action="store_true",
        help="Warn rather than fail if actions are not width 7 or 8",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_calls < 1:
        print("--num-calls must be at least 1", file=sys.stderr)
        return 2
    if args.raw_image_height < 1 or args.raw_image_width < 1:
        print("Raw image dimensions must be positive", file=sys.stderr)
        return 2

    client: WebsocketClientPolicy | None = None
    try:
        host = _normalize_host(args.host)
        if not host:
            raise ValueError("--host must not be empty")

        print(f"Connecting to policy server at {host}:{args.port}...")
        client = WebsocketClientPolicy(host, args.port, timeout=args.timeout)
        server_cfg = client.get_server_metadata()

        print("\nServer metadata:")
        print(json.dumps(server_cfg, indent=2, sort_keys=True, default=_json_default))

        session_id = args.session_id or f"inference-test-{uuid.uuid4()}"
        request_data, request_summary = _build_dummy_request(
            server_cfg,
            prompt=args.prompt,
            session_id=session_id,
            raw_image_height=args.raw_image_height,
            raw_image_width=args.raw_image_width,
            seed=args.seed,
        )

        print("\nDummy request summary:")
        print(json.dumps(request_summary, indent=2, sort_keys=True))

        for call_idx in range(args.num_calls):
            print(f"\nInference call {call_idx + 1}/{args.num_calls}...")
            start = time.monotonic()
            result = client.infer(dict(request_data))
            elapsed_ms = (time.monotonic() - start) * 1000.0
            actions, warnings = _validate_actions(
                result,
                allow_unexpected_action_dim=args.allow_unexpected_action_dim,
            )

            print(
                "Received actions: "
                f"shape={actions.shape}, dtype={actions.dtype}, "
                f"min={actions.min():.4g}, max={actions.max():.4g}, "
                f"latency={elapsed_ms:.1f} ms"
            )
            if result.get("server_timing"):
                print(
                    "Server timing: "
                    + json.dumps(result["server_timing"], sort_keys=True, default=_json_default)
                )
            for warning in warnings:
                print(f"Warning: {warning}")

        print("\nPASS: policy server accepted the RoboArena dummy request and returned actions.")
        return 0

    except Exception as exc:
        print("\nFAIL: policy inference check failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
