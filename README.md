# RoboArena

Welcome to RoboArena, our distributed robot evaluation benchmark! This README explains how to run the **evaluation client** script, how to configure it for your institution’s setup, and tips on providing **high-quality, detailed feedback** that will help us improve our robot learning benchmark.

---

## Installation

1. **Clone or copy** this repository.
2. Ensure you have **Python 3.7+** installed.
3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```
   
4. Make sure **DROID** is installed on your system.

---

## Running the Evaluation

1. Create or edit a YAML config file (similar to `configs/berkeley.yaml`) that contains:
   - `evaluator_name`: Your full name, or some ID you would like to use for evaluators at your university
   - `institution`: Your university
   - `logging_server_ip`: Set this to `34.55.101.123:5000`.
   - `third_person_camera`: The default vantage (e.g. `right_image` or `left_image`).
   - A `cameras` section that identifies the camera `name` and `id` for your institution’s camera setup.

   A minimal example, `my_institution.yaml`, might look like:
   ```yaml
   evaluator_name: John Doe
   institution: Berkeley
   logging_server_ip: 34.55.101.123:5000
   third_person_camera: right_image
   cameras:
     - name: left
       id: 24259877
     - name: right
       id: 24514023
     - name: wrist
       id: 13062452
   ```
   Adjust the IDs to match **your** cameras. Change `third_person_camera` to `left_image` if you prefer the left camera as your default third-person vantage.

3. **Run** the evaluation client script:

   ```bash
   python evaluation_client/main.py configs/my_institution.yaml
   ```

4. **Follow the prompts** in the terminal:
   - Confirm defaults of **evaluator name** and **institution**.
   - Confirm that the left/right cameras are correctly pointing at the part of the scene you want for the third-person view.
   - (Optional) **Switch** between the left or right vantage if you prefer to do so; the script will ask you.
   - **Enter** the **language command** you want the policy to follow (e.g., “pick up the red block and place it in the box”).
   - The system will then run the A/B evaluation (plus additional policies C, D, ...):
     1. **Policy A** rollout
     2. **Policy B** rollout (then it will ask which policy you preferred, A, B, or tie)
     3. **Policy C** rollout
     4. **Policy D** rollout
     5. **...**
   - The script will guide you to provide **partial success** scores.
   - **At the end**, the script asks whether everything went well and if the session should be considered valid. Data from invalid sessions will not be used for our experiments, and we leave this option to handle cases where something went wrong mid-evaluation.

5. **Repeat** as many times as you want. **Between each run** of the entire script:
   - Feel free to **move the robot** to a new location or **change tasks** for the next A/B evaluation. This fosters diverse evaluations. 
   - You can also reposition or switch cameras to create new viewpoints.

---

## Importance of Long-Form Feedback

After you finish evaluating policies A and B, the script prompts for **long-form textual feedback**. This is critical:

1. **Reference Policy A or Policy B** by name. e.g.:
   - “Policy A was a lot faster than policy B, but at the same time exhibited very abrupt movements. Both policies ended up succeeding at the task, but policy B did it more gracefully.”

2. Provide **granular details**:
   - Did one policy complete the task faster/smoother?
   - Could either policy handle highly out-of-distribution instructions or scenes?

---

## Behind the Scenes

- **All data** (videos, partial success ratings, your textual feedback, etc.) is **automatically logged** to the central server at `34.55.101.123:5000`. You do not need to upload anything manually.
- Each evaluation session times out after a while if incomplete, but typically you’ll end the session yourself.

---

## Troubleshooting

**Connection to central server succeeds, but to policy servers fail**: We've observed two instances where university firewalls block IP traffic to bore.pub, the IP forwarding service we are using to host our policy servers. This issue should be on the rarer side, but nevertheless it can be resolved by either (1) connecting to a hotspot instead of university wifi, or (2) setting up a VPN that allows you to bypass the university firewall.

---

**Thank you** for your contributions to this benchmark!

---

