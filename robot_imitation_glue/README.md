## Dataset visualisation:

python lerobot/lerobot/scripts/visualize_dataset_html.py   --root datasets/2025-04-29_15-40-01/ --force-override 1 --repo-id test/test

In the above, the --repo-id doesn't matter, so literally test/test is fine.

## Preprocessing

robot_imitation_glue > ur5station > prepare_datasets.py

Redo visualisation above

## Dataset merging

python lerobot/lerobot/scripts/merge.py --sources datasets_tmp/clothes-hanger-v3p5-EVAL-successes-visionOnly/ datasets_tmp/clothes-hanger-v3p5-rollouts4augmentation-visionOnly/ datasets_tmp/clothes-hanger-v3p6-w500h720-2cam-visionOnly/ --output datasets_tmp/clothes-hanger-v8-visionAugmentedByInstrRollouts --max_dim 7 --fps 10

## Training

python lerobot/lerobot/scripts/train.py --config_path=robot_imitation_glue/ur5station/lerobot_train/...config.json

(extra info in lerobot > examples > advanced > 4_train_policy_with_script)

model size parameters in config: "down_dims", "kernel_size", "diffusion_step_embed_dim",


screen -S session_name -> starts session
ctrl+a ctrl+d -> detaches session
screen -Rd session_name -> reattaches session

## Evaluation

robot_imitation_glue > ur5station > eval_diffusion_lerobot.py