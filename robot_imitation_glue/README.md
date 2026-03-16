## Data collection:
run robot_imitation_glue/robot_imitation_glue/ur5station/data_collection.py

## Upload dataset to HF
hf auth login --token [token] --add-to-git-credential  % for token see jupyterhub script "start_diffusion_training.sh"
hf upload RemkoPr/test_dataset ./datasets/clothes-hanger-v3p7-w500h720-2cam-n50 --repo-type=dataset


## Dataset visualisation:

python lerobot/src/lerobot/scripts/lerobot_dataset_viz.py   --root datasets/2clothes2hanger-test2/ --repo-id test/test --episode-index 0 --display-compressed-images true

OR

Upload your dataset to huggingface, go to https://huggingface.co/spaces/lerobot/visualize_dataset en enter the dataset id.

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

## Downloading HF model
hf download RemkoPr/test_dataset_model --local-dir "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/2026-02-11"

## Evaluation

robot_imitation_glue > ur5station > eval_diffusion_lerobot.py
