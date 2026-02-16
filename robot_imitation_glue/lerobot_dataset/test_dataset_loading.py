from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset = LeRobotDataset(repo_id="whatevs", root="datasets/a-test4")
print(f"Num episodes: {dataset.num_episodes}")
print(dataset[2]['actual_timestamp']-dataset[1]['actual_timestamp'])