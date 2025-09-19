from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from loguru import logger
import numpy as np
import os
from scipy.interpolate import splprep, splev


class ImgCurveDrawer:
    def __init__(self, dataset_root, repo_id, save_dir="curves", episodes_to_exclude=[], failed_episodes=[]):
        self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

        self.failed_episodes = failed_episodes
        self.episodes_to_exclude = episodes_to_exclude
        self.episode_indices_list = self.dataset.episode_data_index
        self.episode_starts = self.episode_indices_list["from"]
        self.num_episodes = len(self.episode_starts)
        logger.debug(f"episode_indices = {self.episode_indices_list}")

        self.current_episode_index = 0
        while self.current_episode_index in self.episodes_to_exclude and self.current_episode_index < self.num_episodes - 1:
            logger.info(f"Skipping episode index {self.current_episode_index} as requested")
            self.current_episode_index += 1
        self.save_dir = dataset_root + "_curves"
        os.makedirs(self.save_dir, exist_ok=True)

        # Matplotlib handles
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.line = None
        self.xs, self.ys = [], []

        # connect events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.onkey)

    def get_image(self, idx):
        """Fetch one dataset image by index"""
        item = dict(self.dataset[int(idx)])
        img = np.array(item["wrist_wilson_image"]) * 255
        img = np.transpose(img, (1, 2, 0))
        return img.astype(np.uint8)

    def show_image(self):
        """Display the current image"""
        img = self.get_image(self.episode_starts[self.current_episode_index])
        self.ax.clear()
        self.ax.imshow(img)
        if self.current_episode_index in self.failed_episodes:
            self.line = Line2D([], [], color="red", linewidth=2)
        else:
            self.line = Line2D([], [], color="green", linewidth=2)
        self.ax.add_line(self.line)
        self.xs, self.ys = [], []
        self.ax.set_title(f"Episode idx {self.current_episode_index} out of {self.num_episodes} total episodes\n"
                          f"Left-click: add point, 'enter': save curve, 'right-click': clear, 'n': next image")
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:  # right-click = remove last point
            if self.xs:
                self.xs.pop()
                self.ys.pop()
        elif event.button == 1:  # left-click = add point
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

        # Update line after adding/removing
        if len(self.xs) > 2:
            from scipy.interpolate import splprep, splev
            tck, u = splprep([self.xs, self.ys], s=0, k=min(3, len(self.xs)-1))
            u_fine = np.linspace(0, 1, 200)
            x_smooth, y_smooth = splev(u_fine, tck)
            self.line.set_data(x_smooth, y_smooth)
        else:
            self.line.set_data(self.xs, self.ys)

        self.fig.canvas.draw()



    def onkey(self, event):
        if event.key == "enter":
            self.save_curve()
        elif event.key == "n":
            self.next_image()

    def save_curve(self):
        """Save the curve only (transparent PNG)"""
        if not self.xs:
            logger.warning("No curve drawn, skipping save")
            return

        if len(self.xs) > 2:
            tck, u = splprep([self.xs, self.ys], s=0, k=min(3, len(self.xs)-1))
            u_fine = np.linspace(0, 1, 200)
            x_smooth, y_smooth = splev(u_fine, tck)
        else:
            x_smooth, y_smooth = self.xs, self.ys

        fig, ax = plt.subplots(figsize=(8, 8))
        if self.current_episode_index in self.failed_episodes:
            ax.plot(x_smooth, y_smooth, color="red", linewidth=2)
        else:
            ax.plot(x_smooth, y_smooth, color="green", linewidth=2)
        ax.axis("off")
        ax.set_xlim(self.ax.get_xlim())
        ax.set_ylim(self.ax.get_ylim())
        outpath = os.path.join(self.save_dir, f"curve_episode{self.current_episode_index:03d}.png")
        fig.savefig(outpath, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        logger.info(f"Saved smooth curve -> {outpath}")


    def next_image(self):
        """Advance to next image and redraw"""
        self.current_episode_index += 1
        while self.current_episode_index in self.episodes_to_exclude and self.current_episode_index < self.num_episodes - 1:
            logger.info(f"Skipping episode index {self.current_episode_index} as requested")
            self.current_episode_index += 1
        if self.current_episode_index >= self.num_episodes:
            logger.info("No more images.")
            plt.close(self.fig)
            return
        self.show_image()

    def run(self):
        self.show_image()
        plt.show()


if __name__ == "__main__":
    '''viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p5-2cam-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[i for i in range(24)] + [i for i in range(30, 35)],
        failed_episodes=[28, 37, 41, 43, 44, 45, 50]
    )
    viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p6-2cam-visionOnly-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[i for i in range(17)],
        failed_episodes=[17, 18, 26, 27, 28, 30, 31, 36, 40, 44, 46]
    )
    viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p8-visionAugmentedByInstrRollouts-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[5],
        failed_episodes=[1, 3, 4, 7, 14, 23, 29, 37, 38, 39]
    )'''
    viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p7-2cam-n50-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[4],
        failed_episodes=[1, 2, 3, 4, 5, 6, 8, 9, 17, 19, 12, 14, 15, 20]
    )
    '''viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p7-2cam-visionOnly-n50-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[],
        failed_episodes=[i for i in range(20) if i not in [2, 11]]
    )
    viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p7-2cam-n100-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[],
        failed_episodes=[0, 3, 4, 6, 8, 9, 12, 13, 14, 15, 17, 18, 19]
    )
    viewer = ImgCurveDrawer(
        dataset_root="datasets/clothes-hanger-v3p7-2cam-visionOnly-n100-EVAL",
        repo_id="repo-id",
        episodes_to_exclude=[],
        failed_episodes=[i for i in range(20) if i not in [0, 5, 6, 15]]
    )'''
    viewer.run()
