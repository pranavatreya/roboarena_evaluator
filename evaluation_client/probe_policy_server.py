"""Probe a RoboArena policy server with dummy observations to characterize
its actual contract vs. the published RoboArena spec.

The RoboArena spec (see base_policy.py docstring) requires these observation keys:
  - observation/joint_position    (7,)
  - observation/cartesian_position (6,)
  - observation/gripper_position  (1,)
  - prompt                        str
  - observation/exterior_image_{i}_left   (H, W, 3)   i in 1..n_external
  - observation/exterior_image_{i}_right  (H, W, 3)   if needs_stereo
  - observation/wrist_image_left          (H, W, 3)   if needs_wrist
  - observation/wrist_image_right         (H, W, 3)   if needs_wrist and needs_stereo
  - session_id                            if needs_session_id

And the server is expected to advertise via get_server_metadata():
  - image_resolution: [H, W]   (clients will resize to this; if absent, client sends raw)
  - n_external_cameras: int
  - needs_wrist_camera: bool
  - needs_stereo_camera: bool
  - needs_session_id: bool
  - action_space: str

Usage:
  python probe_policy_server.py --host <your-policy-host> --port 443
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from websocket_client_policy import WebsocketClientPolicy


DEFAULT_PORT = 443

EXPECTED_METADATA_FIELDS = [
    "image_resolution",
    "n_external_cameras",
    "needs_wrist_camera",
    "needs_stereo_camera",
    "needs_session_id",
    "action_space",
]

CANDIDATE_RESOLUTIONS = [
    (288, 512),   # what extract_observation in main.py downsizes to
    (224, 224),
    (256, 256),
    (480, 640),
    (540, 640),   # what the server's error reported as expected size
    (540, 960),
    (224, 308),
]


# ---------- helpers ----------------------------------------------------------


def _img(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def make_standard_obs(
    image_shape: Tuple[int, int],
    n_external: int,
    needs_wrist: bool,
    needs_stereo: bool,
    include_sid: bool,
    prompt: str = "pick up the red block",
    sid: str = "probe-session",
) -> Dict[str, Any]:
    """RoboArena-spec compliant dummy observation."""
    h, w = image_shape
    obs: Dict[str, Any] = {
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": prompt,
    }
    for i in range(1, n_external + 1):
        obs[f"observation/exterior_image_{i}_left"] = _img(h, w)
        if needs_stereo:
            obs[f"observation/exterior_image_{i}_right"] = _img(h, w)
    if needs_wrist:
        obs["observation/wrist_image_left"] = _img(h, w)
        if needs_stereo:
            obs["observation/wrist_image_right"] = _img(h, w)
    if include_sid:
        obs["session_id"] = sid
    return obs


def _classify_error(msg: str) -> str:
    """Bucket the server-side error so the report is easier to scan."""
    if "KeyError" in msg:
        for line in msg.splitlines():
            if "KeyError" in line:
                return f"missing_key: {line.strip()}"
        return "missing_key"
    if "size" in msg and "tensor" in msg.lower():
        return "shape_mismatch"
    if "RuntimeError" in msg and "size" in msg:
        return "shape_mismatch"
    last = [l for l in msg.strip().splitlines() if l.strip()]
    return f"other: {last[-1] if last else msg}"


def try_infer(host: str, port: int, obs: Dict[str, Any], label: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Open a *fresh* connection for each probe so one failure doesn't poison later tests."""
    print(f"\n--- {label} ---")
    keys = sorted(obs.keys())
    print(f"  sending {len(keys)} keys")
    for k in keys:
        v = obs[k]
        if isinstance(v, np.ndarray):
            print(f"    {k}: ndarray shape={v.shape} dtype={v.dtype}")
        else:
            print(f"    {k}: {type(v).__name__} = {v!r}")
    try:
        client = WebsocketClientPolicy(host, port)
        result = client.infer(obs)
        ret_keys = sorted(result.keys()) if isinstance(result, dict) else None
        print(f"  PASS — server returned keys: {ret_keys}")
        if isinstance(result, dict) and "actions" in result:
            a = np.asarray(result["actions"])
            print(f"    actions: shape={a.shape} dtype={a.dtype} range=[{a.min():.3f}, {a.max():.3f}]")
        return True, result, None
    except Exception as e:
        msg = str(e)
        bucket = _classify_error(msg)
        print(f"  FAIL [{bucket}]")
        for line in msg.strip().splitlines()[-4:]:
            print(f"    {line}")
        return False, None, bucket


# ---------- test cases -------------------------------------------------------


def test_metadata(host: str, port: int) -> Dict[str, Any]:
    print("=" * 70)
    print("TEST 1: get_server_metadata()")
    print("=" * 70)
    client = WebsocketClientPolicy(host, port)
    meta = client.get_server_metadata()
    print("Full metadata returned by server:")
    if isinstance(meta, dict):
        for k, v in meta.items():
            print(f"  {k!r}: {v!r}")
        missing = [f for f in EXPECTED_METADATA_FIELDS if f not in meta]
        extra = [k for k in meta if k not in EXPECTED_METADATA_FIELDS]
        if missing:
            print(f"  WARN — missing RoboArena fields: {missing}")
        if extra:
            print(f"  INFO — extra (non-standard) fields: {extra}")
        if not missing:
            print("  OK — all RoboArena standard metadata fields present.")
    else:
        print(f"  unexpected metadata type: {type(meta)} = {meta!r}")
    return meta if isinstance(meta, dict) else {}


def test_standard_at_declared_shape(host: str, port: int, meta: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("TEST 2: standard obs at metadata-declared image_resolution")
    print("=" * 70)
    n_ext = int(meta.get("n_external_cameras", 1))
    needs_wrist = bool(meta.get("needs_wrist_camera", True))
    needs_stereo = bool(meta.get("needs_stereo_camera", False))
    needs_sid = bool(meta.get("needs_session_id", False))
    img_res = meta.get("image_resolution")

    if not img_res:
        print("  SKIP — server did not advertise image_resolution.")
        return
    h, w = tuple(img_res)
    obs = make_standard_obs((h, w), n_ext, needs_wrist, needs_stereo, needs_sid)
    try_infer(host, port, obs, f"standard obs @ declared {h}x{w}")


def test_standard_at_canonical_288_512(host: str, port: int, meta: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("TEST 3: standard obs at canonical 288x512 (what main.py sends)")
    print("=" * 70)
    n_ext = int(meta.get("n_external_cameras", 1))
    needs_wrist = bool(meta.get("needs_wrist_camera", True))
    needs_stereo = bool(meta.get("needs_stereo_camera", False))
    needs_sid = bool(meta.get("needs_session_id", False))
    obs = make_standard_obs((288, 512), n_ext, needs_wrist, needs_stereo, needs_sid)
    try_infer(host, port, obs, "standard obs @ 288x512")


def test_sweep_resolutions(host: str, port: int, meta: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("TEST 4: sweep common image resolutions (standard keys)")
    print("=" * 70)
    n_ext = int(meta.get("n_external_cameras", 1))
    needs_wrist = bool(meta.get("needs_wrist_camera", True))
    needs_stereo = bool(meta.get("needs_stereo_camera", False))
    needs_sid = bool(meta.get("needs_session_id", False))
    results: List[Tuple[Tuple[int, int], bool, Optional[str]]] = []
    for shape in CANDIDATE_RESOLUTIONS:
        obs = make_standard_obs(shape, n_ext, needs_wrist, needs_stereo, needs_sid)
        ok, _, bucket = try_infer(host, port, obs, f"standard obs @ {shape[0]}x{shape[1]}")
        results.append((shape, ok, bucket))
    print("\nResolution sweep summary:")
    for shape, ok, bucket in results:
        flag = "OK  " if ok else "FAIL"
        print(f"  {flag}  {shape}  {bucket or ''}")


def test_missing_keys(host: str, port: int, meta: Dict[str, Any]) -> None:
    """Drop one image key at a time to confirm what the server actually reads."""
    print("\n" + "=" * 70)
    print("TEST 5: drop one image key at a time (at 540x640)")
    print("=" * 70)
    n_ext = int(meta.get("n_external_cameras", 1))
    needs_wrist = bool(meta.get("needs_wrist_camera", True))
    needs_stereo = bool(meta.get("needs_stereo_camera", False))
    needs_sid = bool(meta.get("needs_session_id", False))

    base = make_standard_obs((540, 640), n_ext, needs_wrist, needs_stereo, needs_sid)
    image_keys = [k for k in base if k.startswith("observation/") and "image" in k]
    for drop in image_keys:
        obs = {k: v for k, v in base.items() if k != drop}
        try_infer(host, port, obs, f"drop {drop}")


# ---------- main -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a RoboArena policy server.")
    parser.add_argument("--host", required=True,
                        help="Policy server host (no scheme, no ws://).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--tests",
        default="metadata,declared,canonical,sweep,drop",
        help="Comma-separated subset of: metadata, declared, canonical, sweep, drop",
    )
    args = parser.parse_args()

    selected = {t.strip() for t in args.tests.split(",")}

    print(f"Target policy server: {args.host}:{args.port}\n")
    meta = test_metadata(args.host, args.port) if "metadata" in selected else {}
    if "declared" in selected:
        test_standard_at_declared_shape(args.host, args.port, meta)
    if "canonical" in selected:
        test_standard_at_canonical_288_512(args.host, args.port, meta)
    if "sweep" in selected:
        test_sweep_resolutions(args.host, args.port, meta)
    if "drop" in selected:
        test_missing_keys(args.host, args.port, meta)

    print("\nDone. Use this report to decide whether the policy server complies with the RoboArena spec.")


if __name__ == "__main__":
    main()
