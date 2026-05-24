"""Hack script: drive a *single* RoboArena-compatible policy server, no central logging.

Bypasses:
  - /version_check, /get_policies_to_compare, /upload_eval_data, /terminate_session
  - A/B preference, partial success, long-form feedback prompts

Keeps:
  - 15 Hz control loop, 400 max timesteps, early-stop on 't'
  - Server metadata negotiation (image_resolution, n_external, needs_wrist, needs_stereo, action_space)
  - Local mp4 dump of left / right / wrist views

Usage:
python evaluation_client/test_single_policy.py configs/my_institution.yaml \
    --host <your-policy-host> --port 443
"""

import argparse
import datetime
import gc
import os
import select
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

try:
    from droid.robot_env import RobotEnv
except ModuleNotFoundError:
    from r2d2.robot_env import RobotEnv

import faulthandler

faulthandler.enable()

from eval_config import EvalConfig, load_config
from websocket_client_policy import WebsocketClientPolicy
import image_tools


DEFAULT_PORT = 443
MAX_TIMESTEPS = 400
CONTROL_HZ = 15
VIDEO_DIR = "./single_policy_videos"


def flush_stdin_buffer() -> None:
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()


def extract_observation(obs_dict: Dict[str, Any], setting: EvalConfig) -> Dict[str, Any]:
    def _is_cam(key: str, cam_name: str) -> bool:
        cam_id = str(setting.cameras.get(cam_name, ""))
        return cam_id and cam_id in key and "left" in key

    def _is_cam_stereo(key: str, cam_name: str) -> bool:
        cam_id = str(setting.cameras.get(cam_name, ""))
        return cam_id and cam_id in key and "right" in key

    img_obs = obs_dict["image"]
    left_img = next((img_obs[k] for k in img_obs if _is_cam(k, "left")), None)
    right_img = next((img_obs[k] for k in img_obs if _is_cam(k, "right")), None)
    wrist_img = next((img_obs[k] for k in img_obs if _is_cam(k, "wrist")), None)
    left_img_stereo = next((img_obs[k] for k in img_obs if _is_cam_stereo(k, "left")), None)
    right_img_stereo = next((img_obs[k] for k in img_obs if _is_cam_stereo(k, "right")), None)
    wrist_img_stereo = next((img_obs[k] for k in img_obs if _is_cam_stereo(k, "wrist")), None)

    def _process(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None:
            return None
        img = img[..., :3]
        img = img[..., ::-1]
        img = np.array(Image.fromarray(img).resize((512, 288), resample=Image.LANCZOS))
        return img

    rs = obs_dict["robot_state"]
    return {
        "left_image": _process(left_img),
        "right_image": _process(right_img),
        "wrist_image": _process(wrist_img),
        "left_image_stereo": _process(left_img_stereo),
        "right_image_stereo": _process(right_img_stereo),
        "wrist_image_stereo": _process(wrist_img_stereo),
        "cartesian_position": np.array(rs["cartesian_position"]),
        "joint_position": np.array(rs["joint_positions"]),
        "gripper_position": np.array([rs["gripper_position"]]),
    }


def run_one_rollout(
    setting: EvalConfig,
    host: str,
    port: int,
    lang_command: str,
    rollout_idx: int,
) -> None:
    base_image = setting.third_person_camera

    print(f"\n=== Rollout #{rollout_idx} — connecting to {host}:{port} ===")
    policy_client = WebsocketClientPolicy(host, port)
    server_cfg: Dict[str, Any] = policy_client.get_server_metadata()
    print(f"Server metadata: {server_cfg}")

    img_res: Optional[Tuple[int, int]] = (
        tuple(server_cfg["image_resolution"]) if server_cfg.get("image_resolution") else None
    )
    needs_wrist = bool(server_cfg.get("needs_wrist_camera", True))
    n_external = int(server_cfg.get("n_external_cameras", 1))
    needs_stereo = bool(server_cfg.get("needs_stereo_camera", False))
    include_sid = bool(server_cfg.get("needs_session_id", False))
    action_space = server_cfg.get("action_space", "joint_position")

    env = RobotEnv(action_space=action_space, gripper_action_space="position")

    print("Type 't' + ENTER at any time to terminate the episode early.\n")

    inference_latencies = []
    frames_left, frames_right, frames_wrist = [], [], []
    pred_action_chunk: Optional[np.ndarray] = None
    actions_from_chunk_completed = 0
    fake_sid = f"single-policy-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    bar = tqdm(range(MAX_TIMESTEPS))
    t_step = 0
    for t_step in bar:
        loop_start = time.time()

        obs = extract_observation(env.get_observation(), setting)

        if obs["left_image"] is not None:
            frames_left.append(obs["left_image"])
        if obs["right_image"] is not None:
            frames_right.append(obs["right_image"])
        if obs["wrist_image"] is not None:
            frames_wrist.append(obs["wrist_image"])

        if (pred_action_chunk is None) or (
            actions_from_chunk_completed >= len(pred_action_chunk)
        ):
            actions_from_chunk_completed = 0

            def _prepare(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if img is None:
                    return None
                if img_res is None:
                    return image_tools.convert_to_uint8(img)
                h, w = img_res
                return image_tools.convert_to_uint8(
                    image_tools.resize(img, h, w, method=Image.LANCZOS)
                )

            request_data: Dict[str, Any] = {
                "observation/joint_position": obs["joint_position"],
                "observation/cartesian_position": obs["cartesian_position"],
                "observation/gripper_position": obs["gripper_position"],
                "prompt": lang_command,
            }

            if n_external == 1:
                request_data["observation/exterior_image_1_left"] = _prepare(obs[base_image])
                if needs_stereo:
                    request_data["observation/exterior_image_1_right"] = _prepare(
                        obs[base_image + "_stereo"]
                    )
            elif n_external == 2:
                request_data["observation/exterior_image_1_left"] = _prepare(obs["left_image"])
                request_data["observation/exterior_image_2_left"] = _prepare(obs["right_image"])
                if needs_stereo:
                    request_data["observation/exterior_image_1_right"] = _prepare(
                        obs["left_image_stereo"]
                    )
                    request_data["observation/exterior_image_2_right"] = _prepare(
                        obs["right_image_stereo"]
                    )

            if needs_wrist and obs["wrist_image"] is not None:
                request_data["observation/wrist_image_left"] = _prepare(obs["wrist_image"])
                if needs_stereo:
                    request_data["observation/wrist_image_right"] = _prepare(
                        obs["wrist_image_stereo"]
                    )

            if include_sid:
                request_data["session_id"] = fake_sid

            infer_t0 = time.time()
            result = policy_client.infer(request_data)
            inference_latencies.append(time.time() - infer_t0)

            pred_action_chunk = np.asarray(result["actions"])
            if pred_action_chunk.ndim == 1:
                pred_action_chunk = pred_action_chunk[None, ...]

        action = np.array(pred_action_chunk[actions_from_chunk_completed], dtype=np.float32)
        actions_from_chunk_completed += 1

        action[-1] = 1.0 if action[-1] > 0.5 else 0.0
        if action_space == "joint_velocity":
            action = np.clip(action, -1, 1)

        env.step(action)

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "t":
                print("Episode terminated early by user.")
                break

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, (1 / CONTROL_HZ) - elapsed))

    flush_stdin_buffer()

    avg_latency = sum(inference_latencies) / len(inference_latencies) if inference_latencies else 0.0
    print(f"\nSteps run: {t_step + 1}  |  avg inference latency: {avg_latency:.3f}s")

    os.makedirs(VIDEO_DIR, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for frames, tag in ((frames_left, "left"), (frames_right, "right"), (frames_wrist, "wrist")):
        if not frames:
            continue
        out_path = os.path.join(VIDEO_DIR, f"rollout{rollout_idx}_{tag}_{ts_str}.mp4")
        ImageSequenceClip(frames, fps=10).write_videofile(
            out_path, codec="libx264", audio=False, logger=None
        )
        print(f"  wrote {out_path}")

    env.reset()
    while (
        input("Did the robot return to its reset pose? (y/n): ").strip().lower() != "y"
    ):
        print("Retrying reset ...")
        env.reset()

    policy_client.reset({"session_id": fake_sid})

    env.close()
    del env
    gc.collect()
    time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a single RoboArena policy server (no logging).")
    parser.add_argument("config_path", type=str, help="Path to evaluator YAML config")
    parser.add_argument("--host", type=str, required=True,
                        help="Policy server host (no scheme, no ws://).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Policy server port. Default: %(default)s")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Language command. If omitted, prompts interactively each rollout.")
    args = parser.parse_args()

    cfg: EvalConfig = load_config(args.config_path)
    print(f"Loaded config: institution={cfg.institution}, third_person={cfg.third_person_camera}")

    rollout_idx = 0
    while True:
        rollout_idx += 1
        flush_stdin_buffer()
        prompt = args.prompt or input("Natural-language command for this rollout: ").strip()

        run_one_rollout(cfg, args.host, args.port, prompt, rollout_idx)

        flush_stdin_buffer()
        again = input("\nRun another rollout? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
