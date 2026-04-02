from robot_imitation_glue.agents.lerobot_agent import make_lerobot_policy_for_inference
import torch


def count_parameters(module):
	total = sum(p.numel() for p in module.parameters())
	trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
	return total, trainable


def print_layer_tree(module, max_depth=3):
	def _print(m, prefix="", depth=0, name="root"):
		total, trainable = count_parameters(m)
		print(f"{prefix}{name}: {m.__class__.__name__}  params={total:,}  trainable={trainable:,}")
		if depth >= max_depth:
			return
		for child_name, child in m.named_children():
			_print(child, prefix + "  ", depth + 1, child_name)

	_print(module)

INCLUDE_INSTR = False
MODEL_NAME = f"b-n200-INSTR{1 if INCLUDE_INSTR else 0}-300k-1enc"
project_root = "/home/rproesma/Documents/Projects/robot_imitation_glue"
checkpoint_path = project_root + f"/outputs/train/{MODEL_NAME}"

policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
# DiffusionPolicy has attribute diffusion [DiffusionModel], which has attribute rgb_encoder [DiffusionRgbEncoder], which has attribute backbone [ResNet, depending on config]
resnet_backbone = policy.diffusion.rgb_encoder.backbone

backbone_file_dir = project_root + "/outputs/vision_backbones/" + f"BACKBONE[{MODEL_NAME}].pth"
#torch.save(resnet_backbone.state_dict(), backbone_file_dir)
resnet_backbone.load_state_dict(torch.load(backbone_file_dir))

print("\n=== Backbone Summary ===")
print(f"model: {MODEL_NAME}")
print(f"checkpoint: {checkpoint_path}")
print(f"backbone_file: {backbone_file_dir}")
print(f"backbone type: {type(resnet_backbone).__name__}")

total_params, trainable_params = count_parameters(resnet_backbone)
print(f"total params: {total_params:,}")
print(f"trainable params: {trainable_params:,}")

print("\n=== Backbone Layer Tree (max_depth=3) ===")
print_layer_tree(resnet_backbone, max_depth=3)

print("\n=== First 40 state_dict keys ===")
for i, key in enumerate(resnet_backbone.state_dict().keys()):
	if i >= 40:
		break
	print(key)