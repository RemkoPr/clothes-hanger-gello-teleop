## Dataset visualisation:

python lerobot/lerobot/scripts/visualize_dataset_html.py   --root datasets/2025-04-29_15-40-01/ --force-override 1 --repo-id test/test

In the above, the --repo-id doesn't matter, so literally test/test is fine.

## Preprocessing

robot_imitation_glue > ur5station > prepare_datasets.py

Redo visualisation above


## Training

python lerobot/lerobot/scripts/train.py --config_path=robot_imitation_glue/ur5station/lerobot_train/...config.json

(extra info in lerobot > examples > advanced > 4_train_policy_with_script)

model size parameters in config: "down_dims", "kernel_size", "diffusion_step_embed_dim",


screen -S session_name -> starts session
ctrl+a ctrl+d -> detaches session
screen -Rd session_name -> reattaches session

## Evaluation

robot_imitation_glue > ur5station > eval_diffusion_lerobot.py