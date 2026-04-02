from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from safetensors.torch import load_file as load_safetensors


DEFAULT_MODELS = [
	"b-n200-INSTR0-200k-1enc-DP0",
	"b-n200-INSTR0-200k-1enc",
	"b-n200-INSTR0-300k-1enc",
	"b-n200-INSTR0-300k",
]


@dataclass
class ModelArtifacts:
	pre_json: dict[str, Any]
	post_json: dict[str, Any]
	pre_state: dict[str, Any]
	post_state: dict[str, Any]


def normalize_json(obj: Any) -> Any:
	if isinstance(obj, dict):
		return {k: normalize_json(v) for k, v in sorted(obj.items())}
	if isinstance(obj, list):
		return [normalize_json(v) for v in obj]
	return obj


def load_json(path: Path) -> dict[str, Any]:
	with path.open("r") as f:
		return json.load(f)


def first_json_diff(a: Any, b: Any, path: str = "root") -> str | None:
	if type(a) is not type(b):
		return f"{path}: type {type(a).__name__} != {type(b).__name__}"

	if isinstance(a, dict):
		a_keys = set(a.keys())
		b_keys = set(b.keys())
		if a_keys != b_keys:
			missing_in_b = sorted(a_keys - b_keys)
			missing_in_a = sorted(b_keys - a_keys)
			return (
				f"{path}: key mismatch | missing_in_b={missing_in_b} | "
				f"missing_in_a={missing_in_a}"
			)
		for key in sorted(a.keys()):
			diff = first_json_diff(a[key], b[key], f"{path}.{key}")
			if diff is not None:
				return diff
		return None

	if isinstance(a, list):
		if len(a) != len(b):
			return f"{path}: list length {len(a)} != {len(b)}"
		for i, (ai, bi) in enumerate(zip(a, b)):
			diff = first_json_diff(ai, bi, f"{path}[{i}]")
			if diff is not None:
				return diff
		return None

	if a != b:
		return f"{path}: value {a!r} != {b!r}"
	return None


def compare_state_dict(a: dict[str, Any], b: dict[str, Any], *, atol: float, rtol: float) -> tuple[bool, dict[str, Any]]:
	keys_a = set(a.keys())
	keys_b = set(b.keys())
	if keys_a != keys_b:
		return False, {
			"missing_in_b": sorted(keys_a - keys_b),
			"missing_in_a": sorted(keys_b - keys_a),
			"value_mismatches": [],
		}

	mismatches: list[tuple[str, str]] = []
	for key in sorted(keys_a):
		tensor_a = a[key]
		tensor_b = b[key]
		if tensor_a.shape != tensor_b.shape:
			mismatches.append((key, f"shape {tuple(tensor_a.shape)} != {tuple(tensor_b.shape)}"))
			continue
		if tensor_a.dtype != tensor_b.dtype:
			mismatches.append((key, f"dtype {tensor_a.dtype} != {tensor_b.dtype}"))
			continue

		equal = bool((tensor_a == tensor_b).all().item())
		if equal:
			continue

		max_abs_diff = float((tensor_a - tensor_b).abs().max().item())
		if max_abs_diff <= atol + rtol * float(tensor_b.abs().max().item()):
			continue
		mismatches.append((key, f"max_abs_diff={max_abs_diff}"))

	return len(mismatches) == 0, {
		"missing_in_b": [],
		"missing_in_a": [],
		"value_mismatches": mismatches,
	}


def require_files(model_dir: Path) -> dict[str, Path]:
	files = {
		"pre_json": model_dir / "policy_preprocessor.json",
		"post_json": model_dir / "policy_postprocessor.json",
		"pre_state": model_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors",
		"post_state": model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
	}
	missing = [str(path) for path in files.values() if not path.exists()]
	if missing:
		raise FileNotFoundError(f"Missing required files for {model_dir.name}: {missing}")
	return files


def load_model_artifacts(model_root: Path, model_name: str) -> ModelArtifacts:
	model_dir = model_root / model_name
	if not model_dir.exists():
		raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

	files = require_files(model_dir)
	pre_json = normalize_json(load_json(files["pre_json"]))
	post_json = normalize_json(load_json(files["post_json"]))
	pre_state = load_safetensors(str(files["pre_state"]))
	post_state = load_safetensors(str(files["post_state"]))
	return ModelArtifacts(pre_json=pre_json, post_json=post_json, pre_state=pre_state, post_state=post_state)


def run_comparison(model_root: Path, models: list[str], *, atol: float, rtol: float) -> bool:
	artifacts = {model: load_model_artifacts(model_root, model) for model in models}

	all_equal = True
	print("Model processor equivalence report")
	print("=" * 88)
	print(f"Model root: {model_root}")
	print(f"Models: {models}")

	for left, right in combinations(models, 2):
		left_artifacts = artifacts[left]
		right_artifacts = artifacts[right]

		pre_json_equal = left_artifacts.pre_json == right_artifacts.pre_json
		post_json_equal = left_artifacts.post_json == right_artifacts.post_json

		pre_json_diff = None if pre_json_equal else first_json_diff(left_artifacts.pre_json, right_artifacts.pre_json)
		post_json_diff = None if post_json_equal else first_json_diff(left_artifacts.post_json, right_artifacts.post_json)

		pre_state_equal, pre_state_details = compare_state_dict(
			left_artifacts.pre_state,
			right_artifacts.pre_state,
			atol=atol,
			rtol=rtol,
		)
		post_state_equal, post_state_details = compare_state_dict(
			left_artifacts.post_state,
			right_artifacts.post_state,
			atol=atol,
			rtol=rtol,
		)

		pair_equal = pre_json_equal and post_json_equal and pre_state_equal and post_state_equal
		all_equal = all_equal and pair_equal

		print(f"\n[{left}] vs [{right}]")
		print(f"  preprocessor json equal:   {pre_json_equal}")
		if pre_json_diff is not None:
			print(f"    first json diff: {pre_json_diff}")

		print(f"  postprocessor json equal:  {post_json_equal}")
		if post_json_diff is not None:
			print(f"    first json diff: {post_json_diff}")

		print(f"  preprocessor state equal:  {pre_state_equal}")
		if not pre_state_equal:
			if pre_state_details["missing_in_b"] or pre_state_details["missing_in_a"]:
				print(
					"    key diffs: "
					f"missing_in_b={pre_state_details['missing_in_b']} | "
					f"missing_in_a={pre_state_details['missing_in_a']}"
				)
			for key, msg in pre_state_details["value_mismatches"][:10]:
				print(f"    mismatch {key}: {msg}")

		print(f"  postprocessor state equal: {post_state_equal}")
		if not post_state_equal:
			if post_state_details["missing_in_b"] or post_state_details["missing_in_a"]:
				print(
					"    key diffs: "
					f"missing_in_b={post_state_details['missing_in_b']} | "
					f"missing_in_a={post_state_details['missing_in_a']}"
				)
			for key, msg in post_state_details["value_mismatches"][:10]:
				print(f"    mismatch {key}: {msg}")

		print(f"  OVERALL processor equivalence: {pair_equal}")

	print("\n" + "=" * 88)
	print(f"All pairwise comparisons equivalent: {all_equal}")
	return all_equal


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Validate pre/post-processor equivalence across model folders.")
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("outputs/train"),
		help="Root directory containing model subfolders.",
	)
	parser.add_argument(
		"--models",
		nargs="+",
		default=DEFAULT_MODELS,
		help="Model folder names to compare pairwise.",
	)
	parser.add_argument(
		"--atol",
		type=float,
		default=0.0,
		help="Absolute tolerance for tensor value comparison.",
	)
	parser.add_argument(
		"--rtol",
		type=float,
		default=0.0,
		help="Relative tolerance for tensor value comparison.",
	)
	parser.add_argument(
		"--fail-on-mismatch",
		action="store_true",
		help="Exit with code 1 when any pair is not equivalent.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	all_equal = run_comparison(args.model_root, args.models, atol=args.atol, rtol=args.rtol)
	if args.fail_on_mismatch and not all_equal:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
