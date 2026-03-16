import torch

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference


INCLUDE_INSTR = True
checkpoint_path = f"/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/b-n200-INSTR{1 if INCLUDE_INSTR else 0}-300k"
    
policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
lerobot_agent = LerobotAgent(policy, preprocessor, postprocessor, "cuda", observation_preprocessor=lambda x: x)
dummy_obs = {
    "observation.images.scene_image": torch.zeros((1, 3, 480, 426)),
    "observation.images.wrist_wilson_image": torch.zeros((1, 3, 480, 426)),
    "observation.state": torch.zeros((1, 11)),  # 6 joints + 1 gripper + 4 clothes_hanger
}

print(lerobot_agent.policy.config.input_features)
print(lerobot_agent.get_action(dummy_obs))
