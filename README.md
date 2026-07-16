# RoboArena Policy Inference Test

This branch is for checking whether a single submitted RoboArena policy server
implements the expected websocket inference API. It does not run a robot, does
not contact the RoboArena central server, and does not create or count a
benchmark evaluation.

Use this before submitting a policy, or when reviewing a newly submitted policy,
to confirm that the policy server:

- accepts the same websocket protocol used by RoboArena evaluations;
- sends valid policy metadata on connection;
- accepts a dummy DROID-style observation payload;
- runs inference on a natural-language prompt; and
- returns a numeric action chunk with a RoboArena-compatible shape.

## Install

Use Python 3.10+.

```bash
git clone <this-repo-url>
cd roboarena_evaluator
git checkout inference_test
pip install -r requirements.txt
```

No DROID, R2D2, robot environment, evaluator code, camera config, or central
server credential is required for this branch.

## Start the Policy Server

Start the submitted policy server exactly as it would be served for RoboArena.
The server must be reachable from the machine running this checker.

The host may be a normal IP address, a DNS name, or a forwarding domain such as
an ngrok hostname. Pasting a host with `ws://`, `wss://`, `http://`, or
`https://` is okay; the checker strips the scheme and first tries
`ws://HOST:PORT`, then falls back to `wss://HOST:PORT`.

## Run the Check

```bash
python evaluation_client/inference_test.py \
  --host roboarena-server.ngrok.com \
  --port 443 \
  --prompt "pick up the red object"
```

For a local server:

```bash
python evaluation_client/inference_test.py --host 127.0.0.1 --port 8000
```

Useful options:

```bash
python evaluation_client/inference_test.py --help
```

Common options:

- `--host`: policy server host or domain.
- `--port`: policy server port.
- `--prompt`: instruction included in the dummy inference request.
- `--num-calls`: number of inference calls to run on one websocket connection.
- `--timeout`: websocket connection timeout in seconds.
- `--session-id`: optional session ID sent only if the server metadata requests
  `needs_session_id`.

## What the Checker Sends

After connecting, the policy server sends metadata such as:

- `image_resolution`
- `needs_wrist_camera`
- `n_external_cameras`
- `needs_stereo_camera`
- `needs_session_id`
- `action_space`

The checker uses that metadata to build the same kind of payload the real
RoboArena evaluator sends:

- `observation/joint_position`: `(7,)` float32
- `observation/cartesian_position`: `(6,)` float32
- `observation/gripper_position`: `(1,)` float32
- `prompt`: string
- requested exterior camera images as `uint8` arrays
- requested wrist camera images as `uint8` arrays
- `session_id`, only if requested by metadata

Images are synthetic and deterministic. If the server declares an
`image_resolution`, the checker resizes the synthetic images to that shape before
sending them, matching the real evaluator behavior.

## Passing Output

A successful run prints the server metadata, a summary of the dummy request, and
the returned action shape, for example:

```text
PASS: policy server accepted the RoboArena dummy request and returned actions.
```

The checker fails if:

- it cannot connect to the websocket server;
- the server does not return metadata;
- inference raises an exception server-side;
- the response does not contain `actions`;
- actions are not numeric, finite, or rank 1/rank 2; or
- the final action dimension is not 7 or 8.

If a policy intentionally returns another action width, use:

```bash
python evaluation_client/inference_test.py \
  --host HOST \
  --port PORT \
  --allow-unexpected-action-dim
```

## Troubleshooting

If connection fails:

- confirm the policy server is running;
- confirm the host and port are reachable from this machine;
- check whether the server expects `ws://` or `wss://`;
- if using a university network, try another network or VPN.

If the server connects but inference fails:

- read the traceback printed by the policy server;
- confirm the policy metadata accurately describes which camera images it needs;
- confirm the policy accepts DROID-style keys such as
  `observation/exterior_image_1_left`, `observation/wrist_image_left`,
  `observation/joint_position`, and `observation/gripper_position`;
- confirm the policy returns a dictionary containing an `actions` array.

This script is only an API and inference sanity check. Passing it does not prove
that a policy is good, safe to run on a robot, or eligible for the public
leaderboard.
