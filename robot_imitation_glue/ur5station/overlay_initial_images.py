from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np


class ImgOverlay:
    def __init__(self, dataset_root, repo_id):
        self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

        self.episode_indices_list = self.dataset.episode_data_index
        self.num_episodes = len(self.episode_indices_list["from"])
        logger.debug(f"episode_indices = {self.episode_indices_list}")

        self.observation_index = 0

        # Matplotlib handles
        self.fig = None
        self.ax_list = []

    def init_plot(self):
        imgs = []
        for start_idx in self.episode_indices_list["from"]:
            item = dict(self.dataset[int(start_idx) + self.observation_index])
            img = np.array(item["wrist_wilson_image"])*255
            img = np.transpose(img, (1, 2, 0))
            imgs.append(img)
        blended_img = (np.mean(imgs, axis=0)).astype(np.uint8)

        #plot blended image
        self.fig, self.ax_list = plt.subplots(1, 1, figsize=(8, 8))
        self.ax_list.imshow(blended_img)


    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    #viewer = ImgOverlay(dataset_root="datasets/clothes-hanger-v3p5-2cam-EVAL", repo_id="repo-id")
    #viewer = ImgOverlay(dataset_root="datasets/clothes-hanger-v3p6-2cam-visionOnly-EVAL", repo_id="repo-id")
    viewer = ImgOverlay(dataset_root="datasets/clothes-hanger-v3p8-visionAugmentedByInstrRollouts-EVAL", repo_id="repo-id")
    

    
    viewer.init_plot()
    viewer.show()
