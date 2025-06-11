## Dataset visualisation:

python lerobot/lerobot/scripts/visualize_dataset_html.py   --root datasets/2025-04-29_15-40-01/ --force-override 1 --repo-id test/test

## Preprocessing

robot_imitation_glue > ur5station > prepare_datasets.py

Redo visualisation above


## Training

python lerobot/lerobot/scripts/train.py --config_path=robot_imitation_glue/ur5station/lerobot_train/...config.json

(extra info in lerobot > examples > advanced > 4_train_policy_with_script)

model size parameters in config: "down_dims", "kernel_size", "diffusion_step_embed_dim",

## Evaluation

robot_imitation_glue > ur5station > eval_diffusion_lerobot.py