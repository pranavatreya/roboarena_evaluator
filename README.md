# CoRL RoboArena Demo

After installing, update the file coex.yaml under configs/ with the camera IDs for your robot setup

There are 3 modes in which RoboArena evaluations can be run:

1. Original mode: python evaluation_client/main.py configs/coex.yaml
2. Demo mode (still A/B, but users can see policy names, and eval data isn't logged to the website): python evaluation_client/demo.py configs/coex.yaml
3. Single-policy evaluation (users will be asked to specify which one policy they want to evaluate): python evaluation_client/single.py configs/coex.yaml
