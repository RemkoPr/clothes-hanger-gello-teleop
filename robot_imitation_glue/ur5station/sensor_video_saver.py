from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib as mpl
mpl.use("TkAgg")  # must be before pyplot import
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import cv2
from scipy.interpolate import interp1d

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 36})
mpl.rcParams['text.usetex'] = True


class SensorVideoSaver:
    def __init__(self, dataset_root, repo_id, episode_idx=0, dataset_type="PREPR"):
        self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

        self.episode_indices_list = self.dataset.episode_data_index
        self.episode_start = self.episode_indices_list["from"][episode_idx]
        self.episode_end = self.episode_indices_list["to"][episode_idx]
        logger.debug(f"episode_indices = {self.episode_indices_list}")
        self.clothes_hanger = []
        self.timestamps = []
        for observation_index in range(self.episode_start, self.episode_end):
            logger.debug(f"Loading observation index {observation_index} (out of {self.episode_start} to {self.episode_end})")
            item = dict(self.dataset[int(observation_index)])
            # collect clothes_hanger vector
            self.clothes_hanger.append(np.array(item['clothes_hanger']))
            # collect timestamp
            self.timestamps.append(float(item['timestamp']))

        logger.debug(f"timestamps = {self.timestamps}")
        self.clothes_hanger = np.array(self.clothes_hanger)  # shape (N, 4)
        self.timestamps = np.array(self.timestamps)
        self.current_index = 0
    

    def run(self, output_path="sensor_values.mp4", fps=30, window_sec=2.0):
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

        # Matplotlib figure setup
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#6595BF", "#B75659", "#D6AE72", "#7DB5A8"]

        # Normalize sensor values once
        initial_values = self.clothes_hanger[0, :]
        normalized = self.clothes_hanger / initial_values

        # Build interpolation functions for each sensor channel
        fns = [
            interp1d(self.timestamps, normalized[:, i], kind="linear", fill_value="extrapolate")
            for i in range(normalized.shape[1])
        ]

        # Define uniform timeline for the video (real-time aligned)
        t_start, t_end = self.timestamps[0], self.timestamps[-1]
        t_uniform = np.arange(t_start, t_end, 1.0 / fps)

        # Fix the x-axis to a static window (last `window_sec` seconds, aligned to 0)
        ax.set_xlim([0, window_sec])
        ax.set_ylim([0.9 * normalized.min(), 1.1 * normalized.max()])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Relative sensor value")

        for t in t_uniform:
            # Sliding window mask
            t0 = t - window_sec
            mask = (t_uniform >= t0) & (t_uniform <= t)
            t_window = t_uniform[mask]

            ax.clear()
            for i, fn in enumerate(fns):
                ax.plot(
                    t_window - t0,   # shift so the window always starts at 0
                    fn(t_window),
                    color=colors[i],
                    label=f"clothes_hanger[{i}]",
                    linewidth=8,
                )

            ax.set_xlim([0, window_sec])
            ax.set_ylim([0.9 * normalized.min(), 1.1 * normalized.max()])
            ax.set_ylabel("Relative\nsensor value")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([0, 0.5, 1], ['0.0', '0.5', '1.0'])

            # Add updating text box showing absolute time
            ax.text(
                0.76, -0.03,
                f"t = {t:.2f} s",
                transform=ax.transAxes,
                fontsize=36,
                verticalalignment="top"
            )

            # Render figure to image
            plt.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize to match video dimensions
            img_bgr = cv2.resize(img_bgr, (1280, 720))
            video_writer.write(img_bgr)

        video_writer.release()
        plt.close(fig)
        print(f"Saved video to {output_path}")





if __name__ == "__main__":
    viewer = SensorVideoSaver(
        dataset_root="datasets/clothes-hanger-v3p5-w500h720-2cam",
        repo_id="repo-id",
        episode_idx=6,
        dataset_type="PREPR"
    )
    viewer.run()
