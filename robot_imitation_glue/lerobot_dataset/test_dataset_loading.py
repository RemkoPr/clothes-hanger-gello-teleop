from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset = LeRobotDataset(repo_id="whatevs", root="datasets/b")
print(f"Num episodes: {dataset.num_episodes}")
#for idx in range(1, len(dataset)):
#    print(dataset[idx]['actual_timestamp']-dataset[idx-1]['actual_timestamp'])
print(dataset[200]['clothes_hanger'])