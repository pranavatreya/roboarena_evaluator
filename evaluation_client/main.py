import argparse
import datetime
import io
import os
import sys
import time
import select
from typing import Tuple, Dict, Any, Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

try:
    from droid.robot_env import RobotEnv
except ModuleNotFoundError:
    from r2d2.robot_env import RobotEnv

import faulthandler

faulthandler.enable()

from eval_config import EvalConfig, load_config
from websocket_client_policy import WebsocketClientPolicy
import image_tools


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #
def flush_stdin_buffer() -> None:
    """Drain anything sitting in stdin so `input()` behaves as expected."""
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()


def extract_observation(obs_dict: Dict[str, Any], setting: EvalConfig) -> Dict[str, Any]:
    """Extract images (left / right / wrist) and robot state from the raw env obs."""

    def _is_cam(key: str, cam_name: str) -> bool:
        cam_id = str(setting.cameras.get(cam_name, ""))
        return cam_id and cam_id in key and "left" in key

    img_obs = obs_dict["image"]
    left_img = next((img_obs[k] for k in img_obs if _is_cam(k, "left")), None)
    right_img = next((img_obs[k] for k in img_obs if _is_cam(k, "right")), None)
    wrist_img = next((img_obs[k] for k in img_obs if _is_cam(k, "wrist")), None)

    def _process(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None:
            return None
        img = img[..., :3]                      # drop alpha
        img = img[..., ::-1]                    # BGR → RGB
        img = np.array(Image.fromarray(img).resize((512, 288), resample=Image.LANCZOS))
        return img

    left_img = _process(left_img)
    right_img = _process(right_img)
    wrist_img = _process(wrist_img)

    rs = obs_dict["robot_state"]
    return {
        "left_image": left_img,
        "right_image": right_img,
        "wrist_image": wrist_img,
        "cartesian_position": np.array(rs["cartesian_position"]),
        "joint_position": np.array(rs["joint_positions"]),
        "gripper_position": np.array([rs["gripper_position"]]),
    }


def check_server_version(server_ip: str) -> None:
    """Abort if the central server and client are out-of-sync."""
    url = f"http://{server_ip}/version_check"
    payload = {"client_version": "1.1"}
    try:
        r = requests.post(url, json=payload)
        if not r.ok:
            print(
                "⚠️  Version mismatch with central server. "
                "Please pull the latest evaluation client."
            )
            sys.exit(1)
    except Exception as e:
        print(f"Failed version check – server unreachable?\n{e}")
        sys.exit(1)


# --------------------------------------------------------------------------- #
#  Main evaluation routine                                                    #
# --------------------------------------------------------------------------- #
def run_evaluation(setting: EvalConfig, evaluator_email: str, institution: str) -> None:
    """Main evaluation loop – runs through all policies returned by the server."""

    # ----------------------------------------------------------------------- #
    #  Handshake with central server                                          #
    # ----------------------------------------------------------------------- #
    check_server_version(setting.logging_server_ip)

    # Temporary env just for camera alignment prompt
    env_preview = RobotEnv(action_space="joint_position", gripper_action_space="position")
    preview_obs = extract_observation(env_preview.get_observation(), setting)

    left_img = preview_obs["left_image"] if preview_obs["left_image" is not None else np.zeros((288, 512, 3), dtype=np.uint8)
    right_img = preview_obs["right_image"] if preview_obs["right_image"] is not None else np.zeros((288, 512, 3), dtype=np.uint8)
    preview_concat = np.concatenate([left_img, right_img], axis=1)

    plt.imshow(preview_concat)
    plt.title(
        "Current left / right third-person views.\n"
        "Ensure cameras are pointed at the robot & scene."
    )
    plt.show(block=False)

    if input("Are these cameras positioned correctly? (y/n): ").strip().lower() != "y":
        print("Exiting – please adjust cameras and restart.")
        plt.close()
        sys.exit(0)
    plt.close()

    base_image = setting.third_person_camera  # default vantage (left_image | right_image)
    if (
        input(
            f"Default third-person camera is “{base_image}”. "
            "Type “switch” to swap or press ENTER to keep: "
        )
        .strip()
        .lower()
        == "switch"
    ):
        base_image = "right_image" if base_image == "left_image" else "left_image"

    # ----------------------------------------------------------------------- #
    #  Request policy list                                                    #
    # ----------------------------------------------------------------------- #
    resp = requests.get(
        f"http://{setting.logging_server_ip}/get_policies_to_compare",
        params={
            "eval_location": institution,
            "evaluator_name": evaluator_email, # email is the primary form of id now
            "robot_name": "DROID",
        },
    )
    if not resp.ok:
        print("Failed to obtain policies from central server:")
        print(resp.status_code, resp.text)
        sys.exit(1)

    session_info = resp.json()
    session_id: str = session_info["session_id"]
    policies: list[Dict[str, Any]] = session_info["policies"]

    print(
        f"\n✅  Session started (id = {session_id}). "
        "We’ll evaluate policies A then B in sequence.\n"
    )

    lang_command = input("Natural-language command to send to both policies: ")

    # Save a reference state for “reset scene” prompt later
    ref_reset_state = preview_concat.copy()
    env_preview.close()

    preference_ab: Optional[str] = None
    comparative_feedback: Optional[str] = None
    max_timesteps = 400

    # ----------------------------------------------------------------------- #
    #  Iterate through policies A and B.                                      #
    # ----------------------------------------------------------------------- #
    for i, pol in enumerate(policies):
        label, ip, port = pol["label"], pol["ip"], pol["port"]

        print(f"\n=== Evaluating policy {label} ===")
        print("ℹ️  Type 't' + ENTER at any time to terminate the episode early.\n")

        # 1. Connect to the policy server and grab its declared config
        policy_client = WebsocketClientPolicy(ip, port)
        server_cfg: Dict[str, Any] = policy_client.get_server_metadata()

        # Make sure we interpret resolution correctly
        img_res: Optional[Tuple[int, int]] = tuple(server_cfg["image_resolution"]) \
            if server_cfg.get("image_resolution") else None
        needs_wrist = bool(server_cfg.get("needs_wrist_camera", True))
        n_external = int(server_cfg.get("n_external_cameras", 1))
        include_sid = bool(server_cfg.get("needs_session_id", False))
        action_space = server_cfg.get("action_space", "joint_position")

        # 2. New RobotEnv instance as requested
        env = RobotEnv(action_space=action_space, gripper_action_space="position")

        # 3. Let the server (policy) reset internal state if desired
        policy_client.reset()

        # 4. Buffers for logging & video
        inference_latencies: list[float] = []
        frames_left, frames_right, frames_wrist = [], [], []
        episode_data = []

        pred_action_chunk: Optional[np.ndarray] = None
        actions_from_chunk_completed = 0

        bar = tqdm(range(max_timesteps))
        for t_step in bar:
            loop_start = time.time()

            raw_obs = env.get_observation()
            obs = extract_observation(raw_obs, setting)

            # Video logging
            if obs["left_image"] is not None:
                frames_left.append(obs["left_image"])
            if obs["right_image"] is not None:
                frames_right.append(obs["right_image"])
            if obs["wrist_image"] is not None:
                frames_wrist.append(obs["wrist_image"])

            # -----------------------------------------------------------------
            #  Fetch a new open-loop action chunk if needed
            # -----------------------------------------------------------------
            if (pred_action_chunk is None) or (
                actions_from_chunk_completed >= len(pred_action_chunk)
            ):
                actions_from_chunk_completed = 0

                def _prepare(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
                    if img is None:
                        return None
                    if img_res is None:  # server wants raw
                        return image_tools.convert_to_uint8(img)
                    h, w = img_res
                    return image_tools.convert_to_uint8(
                        image_tools.resize(img, h, w, method=Image.LANCZOS)
                    )

                request_data: Dict[str, Any] = {
                    # Joint / gripper are always sent
                    "observation/joint_position": obs["joint_position"],
                    "observation/gripper_position": obs["gripper_position"],
                    "prompt": lang_command,
                }

                # Include third-person images
                if n_external == 1:
                    request_data["observation/exterior_image_1_left"] = _prepare(
                        obs[base_image]
                    )
                elif n_external == 2:
                    request_data["observation/exterior_image_1_left"] = _prepare(
                        obs["left_image"]
                    )
                    request_data["observation/exterior_image_2_left"] = _prepare(
                        obs["right_image"]
                    )

                # Wrist camera (optional)
                if needs_wrist and obs["wrist_image"] is not None:
                    request_data["observation/wrist_image_left"] = _prepare(
                        obs["wrist_image"]
                    )

                # Session id (optional)
                if include_sid:
                    request_data["session_id"] = session_id

                # Inference
                infer_t0 = time.time()
                result = policy_client.infer(request_data)
                inference_latencies.append(time.time() - infer_t0)

                pred_action_chunk = np.asarray(result["actions"])
                if pred_action_chunk.ndim == 1:
                    pred_action_chunk = pred_action_chunk[None, ...]

            # -----------------------------------------------------------------
            #  Execute one low-level action
            # -----------------------------------------------------------------
            action = np.array(
                pred_action_chunk[actions_from_chunk_completed], dtype=np.float32
            )
            actions_from_chunk_completed += 1

            # Binarise gripper open / close
            action[-1] = 1.0 if action[-1] > 0.5 else 0.0
            
            # Only clip action if using joint_velocity action space
            if action_space == "joint_velocity":
                action = np.clip(action, -1, 1)

            env.step(action)

            # Log step
            episode_data.append(
                {
                    "cartesian_position": obs["cartesian_position"].tolist(),
                    "joint_position": obs["joint_position"].tolist(),
                    "gripper_position": obs["gripper_position"].tolist(),
                    "action": action.tolist(),
                }
            )

            # Early-terminate if user presses 't'
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "t":
                    print("⛔  Episode terminated early by user.")
                    break

            # Maintain ~15 Hz control rate
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1 / 15) - elapsed))

        flush_stdin_buffer()

        # ---------------------------------------------------------------------#
        #  User feedback (success & preference)                                 #
        # ---------------------------------------------------------------------#
        while True:
            try:
                partial_succ = float(
                    input(f"Rate partial success of policy {label} (0-100): ")
                )
                if 0.0 <= partial_succ <= 100.0:
                    partial_succ /= 100.0
                    break
            except ValueError:
                pass
            print("Please enter a number between 0 and 100.")

        bin_succ = 1 if partial_succ == 1.0 else 0

        if label == "B":
            flush_stdin_buffer()
            while True:
                pref = input("Which policy did you prefer, A, B, or 'tie'? ").strip().lower()
                if pref in ["a", "b", "tie"]:
                    preference_ab = pref.upper()
                    break
                print("Please enter 'A', 'B', or 'tie' exactly.")
            flush_stdin_buffer()
            comparative_feedback = input(
                "Now please provide long-form textual feedback comparing policy A vs. policy B:\n"
            )
            flush_stdin_buffer()
            comparative_feedback = comparative_feedback.strip() # remove any starting or ending whitespace
            while True:
                print()
                print("Thanks for entering long-form feedback! This is the feedback you gave:\n")
                print("###############################################")
                print(comparative_feedback)
                print("###############################################\n")
                should_move_on = input("If this looks good, hit 'y' to move on, otherwise hit 'n' and we'll give you a chance to enter feedback again: ")
                flush_stdin_buffer()
                if should_move_on.strip().lower() == 'y':
                    break
                comparative_feedback = input(
                    "Please provide long-form textual feedback comparing policy A vs. policy B:\n"
                )
                flush_stdin_buffer()

        # ---------------------------------------------------------------------#
        #  Save & upload episode artifacts                                      #
        # ---------------------------------------------------------------------#
        print()
        ts_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        def _encode_video(frames: list[np.ndarray], tag: str):
            if not frames:
                return None
            tmp = f"/tmp/temp_{tag}.mp4"
            ImageSequenceClip(frames, fps=10).write_videofile(
                tmp, codec="libx264", audio=False, verbose=False, logger=None
            )
            with open(tmp, "rb") as f_:
                data = f_.read()
            os.remove(tmp)
            return tag, (f"{tag}.mp4", data, "video/mp4")

        files = {}
        for res in (
            _encode_video(frames_left, "left"),
            _encode_video(frames_right, "right"),
            _encode_video(frames_wrist, "wrist"),
        ):
            if res:
                files[f"video_{res[0]}"] = res[1]

        npz_tmp = "/tmp/temp_episode_data.npz"
        np.savez_compressed(npz_tmp, data=episode_data)
        with open(npz_tmp, "rb") as f_npz:
            files["npz_file"] = ("episode_data.npz", f_npz.read(), "application/octet-stream")
        os.remove(npz_tmp)

        avg_latency = (
            sum(inference_latencies) / len(inference_latencies) if inference_latencies else 0.0
        )
        policy_letter_with_latency = f"{label};avg_latency={avg_latency:.3f}"

        data = {
            "session_id": session_id,
            "command": lang_command,
            "binary_success": str(bin_succ),
            "partial_success": f"{partial_succ:.3f}",
            "duration": str(t_step),
            "policy_ip": str(ip),
            "policy_port": str(port),
            "third_person_camera_type": base_image,
            "third_person_camera_id": str(setting.cameras.get(base_image, "")),
            "policy_letter": policy_letter_with_latency,
            "timestamp": ts_str,
        }

        print("⬆️  Uploading episode data …")
        up_resp = requests.post(
            f"http://{setting.logging_server_ip}/upload_eval_data", files=files, data=data
        )
        if not up_resp.ok:
            print("❌  Upload failed:", up_resp.text)
            sys.exit(1)
        print("✅  Upload succeeded.")

        # ---------------------------------------------------------------------#
        #  Reset robot & scene                                                 #
        # ---------------------------------------------------------------------#
        env.reset()
        while (
            input("Did the robot return to its reset pose? Sometimes it may fail to do so (y/n): ")
            .strip()
            .lower()
            != "y"
        ):
            print("Retrying reset …")
            env.reset()

        if i < len(policies) - 1:
            fig, ax = plt.subplots()
            ax.set_title("Reminder: reset scene to the original starting condition.\nFor reference, this is what your starting state looked like:")
            ax.imshow(ref_reset_state)
            plt.show(block=False)
            input("Press ENTER once the scene has been reset.")
            plt.close(fig)

        env.close()

    # -------------------------------------------------------------------------#
    #  Session termination                                                     #
    # -------------------------------------------------------------------------#
    valid = input("Mark this session as **valid**? (y/n): ").strip().lower() == "y"
    notes = ""
    if valid:
        notes += "VALID_SESSION:\n"
    if preference_ab:
        notes += f"PREFERENCE={preference_ab}\n"
    if comparative_feedback:
        notes += f"LONGFORM_FEEDBACK={comparative_feedback}\n"

    requests.post(
        f"http://{setting.logging_server_ip}/terminate_session",
        data={"session_id": session_id, "evaluation_notes": notes},
    )

    print(
        f"\n🎉  Evaluation session {session_id} complete – thank you!\n"
        "(If the script hasn’t exited automatically, press Ctrl-C.)"
    )
    sys.exit(0)


# --------------------------------------------------------------------------- #
#  Entrypoint                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RoboArena evaluation session.")
    parser.add_argument("config_path", type=str, help="Path to the YAML config")
    args = parser.parse_args()

    cfg: EvalConfig = load_config(args.config_path)

    default_email = cfg.evaluator_email
    default_inst = cfg.institution

    if default_email and default_inst:
        print(
            f"Config file specifies evaluator_email = {default_email}, "
            f"institution = {default_inst}."
        )
        if input("Press ENTER to accept, or type 'change' to override: ").strip().lower():
            default_email = None
            default_inst = None

    if not default_email:
        default_email = input("Evaluator email: ").strip()
    if not default_inst:
        default_inst = input("Institution (e.g. Berkeley, UPenn): ").strip()

    run_evaluation(cfg, default_email, default_inst)
