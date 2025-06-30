# RoboArena

Thanks for being a part of the evaluation team of RoboArena! This README explains how to run the evaluation client script, how to configure it for your institution’s setup, and some tips on picking tasks for evaluation and providing high-quality feedback used for evaluating policies.

---

## Evaluator Guide

We wrote an evaluate guide here https://docs.google.com/document/d/1g-Z1lGUozbynA3U2khogBdLBZnrRTg9o2oqZUZkOwI4/edit?usp=sharing. Please make sure you're faimilar with its contents before running evals!

## Installation

1. Clone this repo.
2. Ensure you have **Python 3.7+** installed.
3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```
   
4. Make sure **DROID** is installed on your system.

---

## Running the Evaluation

1. Create or edit a YAML config file (similar to `configs/berkeley.yaml`) that contains:
   - `evaluator_email`: Your email. Email will be the primary form of identification for all evaluators and submitters. Make sure the same email is used for evaluations and for policy submissions, so you can get extra eval credit for evaluating policies. 
   - `institution`: Your university/institution
   - `logging_server_ip`: Should be `34.55.101.123:5000`.
   - `third_person_camera`: The default vantage point (e.g. `right_image` or `left_image`).
   - A `cameras` section that identifies the camera `name` and `id` for your institution’s camera setup.

   A minimal example, `my_institution.yaml`, might look like:
   ```yaml
   evaluator_email: oski_bear@gmail.com
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
   Make sure to adjust the camera IDs to match your setup. See https://droid-dataset.github.io/droid/ for more details.

3. **Run** the evaluation client script:

   ```bash
   python evaluation_client/main.py configs/my_institution.yaml
   ```

4. **Follow the prompts** in the terminal:
   - Confirm defaults of evaluator email and institution.
   - Confirm that the left/right cameras are correctly pointing at the part of the scene you want for the third-person view.
   - (Optional) Switch between the left or right vantage if you prefer to do so; the script will ask you.
   - Enter the language command you want the policy to follow (e.g., “pick up the red block and place it in the box”).
   - The system will then run the A/B evaluation
     1. **Policy A** rollout (it will then ask for partial success)
     2. **Policy B** rollout (it will ask for partial success, then it will ask which policy you preferred, A, B, or tie)
   - **Important**: there are 3 types of feedback the script asks you to provide: partial success, A/B preference, and long-form feedback. For partial success please give your best approximate guess -- it's up to you how you assign partial credit points. The most important form of feedback is the next one, A/B preference. This is used to construct the global policy rankings. For long-form feedback, see additional instructions below.
   - **At the end**, the script asks whether everything went well and if the session should be considered valid. Data from invalid sessions will not be used for our experiments, and we leave this option to handle cases where something went wrong mid-evaluation.

6. **Repeat** as many times as you want. **Between each run** of the entire script:
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

