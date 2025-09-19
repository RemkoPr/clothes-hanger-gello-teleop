from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from loguru import logger
import numpy as np
import cv2

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
mpl.rcParams['text.usetex'] = True


class FrameSaver:
    def __init__(self, dataset_root, repo_id, episode_idx=0, dataset_type="PREPR"):
        self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

        self.episode_indices_list = self.dataset.episode_data_index
        self.episode_start = self.episode_indices_list["from"][episode_idx]
        self.episode_end = self.episode_indices_list["to"][episode_idx]
        logger.debug(f"episode_indices = {self.episode_indices_list}")

        # Load all frames + clothes_hanger values for the episode
        self.frames = {'scene': [], 'wrist': []}
        self.clothes_hanger = []
        self.timestamps = []
        for observation_index in range(self.episode_start, self.episode_end):
            item = dict(self.dataset[int(observation_index)])
            if dataset_type == "PREPR":
                scene_img = np.array(item['observation.images.scene_image'])
                scene_img = np.transpose(scene_img, (1, 2, 0))
                wrist_img = np.array(item['observation.images.wrist_wilson_image'])
                wrist_img = np.transpose(wrist_img, (1, 2, 0))
            elif dataset_type == "EVAL":
                scene_img = self.preprocessor(np.array(item['scene_image']))
                wrist_img = self.preprocessor(np.array(item['wrist_wilson_image']))
            self.frames['scene'].append(scene_img)
            self.frames['wrist'].append(wrist_img)

            # collect clothes_hanger vector
            self.clothes_hanger.append(np.array(item['clothes_hanger']))

            # collect timestamp
            self.timestamps.append(item['timestamp'])

        logger.debug(f"timestamps = {self.timestamps}")
        self.clothes_hanger = np.array(self.clothes_hanger)  # shape (N, 4)
        self.current_index = 0


    def run(self):
        # ========== Window 1: Frame viewer ==========
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.im = self.ax.imshow(self.frames['scene'][int(self.current_index)])
        self.ax.set_title(f"Frame {self.current_index}")

        # Slider for frame index
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, "Index", 0, len(self.frames['scene'])-1, valinit=0, valstep=1)

        def update(val):
            self.current_index = int(self.slider.val)
            self.im.set_data(self.frames['scene'][self.current_index])
            self.ax.set_title(f"Frame {self.current_index}")
            self.fig.canvas.draw_idle()

        self.slider.on_changed(update)

        # Button to save current frame
        ax_button = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(ax_button, "Save")

        def save_frame(event):
            current_timestamp = float(self.timestamps[self.current_index])

            # format: integer if large, or fixed 3 decimals if float
            ts_str = f"{current_timestamp:.1f}".replace(".", "p")

            fname = f"idx_{self.current_index}_{ts_str}.png"
            plt.imsave('scene_' + fname, self.frames['scene'][self.current_index])
            plt.imsave('wrist_' + fname, self.frames['wrist'][self.current_index])
            logger.info(f"Saved frame {self.current_index} at {current_timestamp} as {fname}")


        self.button.on_clicked(save_frame)

        # ========== Window 2: Clothes hanger plot ==========
        fig2, ax2 = plt.subplots()
        colors = ["#6595BF", "#B75659", "#D6AE72", "#7DB5A8"]

        # normalize each sensor trace by its initial value
        initial_values = self.clothes_hanger[0, :]
        for i in range(self.clothes_hanger.shape[1]):
            normalized = self.clothes_hanger[:, i] / initial_values[i]
            ax2.plot(
                self.timestamps,
                normalized,
                color=colors[i],
                label=f"clothes_hanger[{i}]",
                linewidth=3,
            )

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Relative sensor value")
        # Show both figures
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    viewer = FrameSaver(
        dataset_root="datasets/clothes-hanger-v3p5-w500h720-2cam",
        repo_id="repo-id",
        episode_idx=6
    )
    viewer.run()
