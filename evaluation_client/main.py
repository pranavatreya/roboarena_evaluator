import argparse
import datetime
import io
import os
import sys
import time
import select
from typing import Tuple, Dict, Any, Optional
import gc

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

from eval_config import EvalConfig, load_config, save_evaluator_code
from websocket_client_policy import WebsocketClientPolicy
import image_tools

CLIENT_VERSION = "1.5"
POLICY_PAIR_REQUEST_TIMEOUT_SECS = 120


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
        img = img[..., :3]                      # drop alpha
        img = img[..., ::-1]                    # BGR → RGB
        img = np.array(Image.fromarray(img).resize((512, 288), resample=Image.LANCZOS))
        return img

    left_img = _process(left_img)
    right_img = _process(right_img)
    wrist_img = _process(wrist_img)
    left_img_stereo = _process(left_img_stereo)
    right_img_stereo = _process(right_img_stereo)
    wrist_img_stereo = _process(wrist_img_stereo)

    rs = obs_dict["robot_state"]
    return {
        "left_image": left_img,
        "right_image": right_img,
        "wrist_image": wrist_img,
        "left_image_stereo": left_img_stereo,
        "right_image_stereo": right_img_stereo,
        "wrist_image_stereo": wrist_img_stereo,
        "cartesian_position": np.array(rs["cartesian_position"]),
        "joint_position": np.array(rs["joint_positions"]),
        "gripper_position": np.array([rs["gripper_position"]]),
    }


def check_server_version(server_ip: str) -> None:
    """Abort if the central server and client are out-of-sync."""
    url = f"http://{server_ip}/version_check"
    payload = {"client_version": CLIENT_VERSION}
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


def validate_evaluator_access(
    server_ip: str,
    evaluator_email: str,
    institution: str,
    evaluator_code: str,
) -> None:
    """Abort if the evaluator access code is missing or rejected."""
    url = f"http://{server_ip}/validate_evaluator_access"
    payload = {
        "evaluator_email": evaluator_email,
        "institution": institution,
        "evaluator_code": evaluator_code,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if not r.ok:
            print("Evaluator access code was rejected by the central server:")
            print(r.status_code, r.text)
            sys.exit(1)
    except Exception as e:
        print(f"Failed evaluator access-code check – server unreachable?\n{e}")
        sys.exit(1)


# --------------------------------------------------------------------------- #
#  Main evaluation routine                                                    #
# --------------------------------------------------------------------------- #
def run_evaluation(
    setting: EvalConfig,
    evaluator_email: str,
    institution: str,
    evaluator_code: str,
    config_path: str,
) -> None:
    """Main evaluation loop – runs through all policies returned by the server."""

    # ----------------------------------------------------------------------- #
    #  Handshake with central server                                          #
    # ----------------------------------------------------------------------- #
    check_server_version(setting.logging_server_ip)
    validate_evaluator_access(
        setting.logging_server_ip,
        evaluator_email,
        institution,
        evaluator_code,
    )
    save_evaluator_code(config_path, evaluator_code)

    # Temporary env just for camera alignment prompt
    env_preview = RobotEnv(action_space="joint_position", gripper_action_space="position")
    preview_obs = extract_observation(env_preview.get_observation(), setting)

    left_img = preview_obs["left_image"] if preview_obs["left_image"] is not None else np.zeros((288, 512, 3), dtype=np.uint8)
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

    lang_command = input("Natural-language command to send to both policies: ").strip()
    while not lang_command:
        print("Please enter a non-empty natural-language command before policies are assigned.")
        lang_command = input("Natural-language command to send to both policies: ").strip()

    # ----------------------------------------------------------------------- #
    #  Request policy list after the task has been fixed                      #
    # ----------------------------------------------------------------------- #
    print(
        "\nRequesting an A/B policy pair from the central server. "
        "This can take a little while if policy availability is being checked; "
        "please wait...",
        flush=True,
    )
    try:
        resp = requests.post(
            f"http://{setting.logging_server_ip}/get_policies_to_compare",
            json={
                "eval_location": institution,
                "evaluator_name": evaluator_email, # email is the primary form of id now
                "evaluator_code": evaluator_code,
                "language_instruction": lang_command,
                "robot_name": "DROID",
            },
            timeout=POLICY_PAIR_REQUEST_TIMEOUT_SECS,
        )
    except requests.exceptions.Timeout:
        print(
            "Timed out while waiting for the central server to assign policies. "
            "Please try again in a few minutes."
        )
        sys.exit(1)
    except requests.RequestException as e:
        print("Failed to contact the central server while assigning policies:")
        print(e)
        sys.exit(1)

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

    # Save a reference state for “reset scene” prompt later
    ref_reset_state = preview_concat.copy()
    env_preview.close()
    del env_preview
    gc.collect()
    time.sleep(0.2)

    preference_ab: Optional[str] = None
    comparative_feedback: Optional[str] = None
    max_timesteps = 400

    # ----------------------------------------------------------------------- #
    #  Iterate through policies A and B.                                      #
    # ----------------------------------------------------------------------- #
    for i, pol in enumerate(policies):
        label, ip, port = pol["label"], pol["ip"], pol["port"]

        # Create a second, policy specific session ID, for passing to policy client
        session_id_x_policy = session_id + "-" + label

        print(f"Setting up the robot environment for policy {label}...")

        # 1. Connect to the policy server and grab its declared config
        policy_client = WebsocketClientPolicy(ip, port)
        server_cfg: Dict[str, Any] = policy_client.get_server_metadata()

        # Make sure we interpret resolution correctly
        img_res: Optional[Tuple[int, int]] = tuple(server_cfg["image_resolution"]) \
            if server_cfg.get("image_resolution") else None
        needs_wrist = bool(server_cfg.get("needs_wrist_camera", True))
        n_external = int(server_cfg.get("n_external_cameras", 1))
        needs_stereo = bool(server_cfg.get("needs_stereo_camera", False))
        include_sid = bool(server_cfg.get("needs_session_id", False))
        action_space = server_cfg.get("action_space", "joint_position")

        # If policy server requested two wrist images, but local robot setup only supports one, 
        # notify evaluator and terminate.
        left_exists_on_robot = str(setting.cameras.get("left", "")) != ""
        right_exists_on_robot = str(setting.cameras.get("right", "")) != ""
        both_exist = left_exists_on_robot and right_exists_on_robot
        if n_external == 2:
            assert both_exist, f"Policy {label} requested images from two third-person cameras, but your robot setup only has one. The script will now terminate because this policy cannot be evaluated. Please run this script again, and it's possible that the next time, the policies you will be given to evaluate only request one external camera. We understand this isn't the most optimal solution, and we'll work in the future to make this more seamless for you. Please contact us if this proves a significant inconvenience."
            if input("This next policy you are about to evaluate needs 2 third-person (i.e., shoulder) cameras. Does your setup have 2 third-person cameras? (y/n): ").strip().lower() != "y":
                print(f"Policy {label} requested images from two third-person cameras, but your robot setup only has one. The script will now terminate because this policy cannot be evaluated. Please run this script again, and it's possible that the next time, the policies you will be given to evaluate only request one external camera. We understand this isn't the most optimal solution, and we'll work in the future to make this more seamless for you. Please contact us if this proves a significant inconvenience.")
                sys.exit(0)

        # 2. New RobotEnv instance as requested
        env = RobotEnv(action_space=action_space, gripper_action_space="position")

        print(f"\n=== Evaluating policy {label} ===")
        print("ℹ️  Type 't' + ENTER at any time to terminate the episode early.\n")

        # 3. Buffers for logging & video
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
                    # Joint / cartesian / gripper are always sent
                    "observation/joint_position": obs["joint_position"],
                    "observation/cartesian_position": obs["cartesian_position"],
                    "observation/gripper_position": obs["gripper_position"],
                    "prompt": lang_command,
                }

                # Include third-person images
                if n_external == 1:
                    request_data["observation/exterior_image_1_left"] = _prepare(
                        obs[base_image]
                    )
                    if needs_stereo:
                        request_data["observation/exterior_image_1_right"] = _prepare(
                            obs[base_image + "_stereo"]
                        )
                elif n_external == 2:
                    request_data["observation/exterior_image_1_left"] = _prepare(
                        obs["left_image"]
                    )
                    request_data["observation/exterior_image_2_left"] = _prepare(
                        obs["right_image"]
                    )
                    if needs_stereo:
                        request_data["observation/exterior_image_1_right"] = _prepare(
                            obs["left_image_stereo"]
                        )
                        request_data["observation/exterior_image_2_right"] = _prepare(
                            obs["right_image_stereo"]
                        )  

                # Wrist camera (optional)
                if needs_wrist and obs["wrist_image"] is not None:
                    request_data["observation/wrist_image_left"] = _prepare(
                        obs["wrist_image"]
                    )
                    if needs_stereo:
                        request_data["observation/wrist_image_right"] = _prepare(
                            obs["wrist_image_stereo"]
                        ) 

                # Session id (optional)
                if include_sid:
                    request_data["session_id"] = session_id_x_policy

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

        def _encode_video(frames, tag):
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
            "evaluator_code": evaluator_code,
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

        # Call reset() on policy client
        reset_info = {"session_id": session_id_x_policy}
        policy_client.reset(reset_info)

        if i < len(policies) - 1:
            fig, ax = plt.subplots()
            ax.set_title("Reminder: reset scene to the original starting condition.\nFor reference, this is what your starting state looked like:")
            ax.imshow(ref_reset_state)
            plt.show(block=False)
            input("Press ENTER once the scene has been reset.")
            plt.close(fig)

        # Close and garbage collect the env
        env.close()
        del env
        gc.collect()
        time.sleep(0.2)

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
        data={
            "session_id": session_id,
            "evaluator_code": evaluator_code,
            "evaluation_notes": notes,
        },
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
    default_code = cfg.evaluator_code

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
    if not default_code:
        default_code = input("Evaluator access code: ").strip()
    while not default_code:
        print("An evaluator access code is required. Ask the RoboArena team for one.")
        default_code = input("Evaluator access code: ").strip()

    run_evaluation(cfg, default_email.strip().lower(), default_inst, default_code, args.config_path)
