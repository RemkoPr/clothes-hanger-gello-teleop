from cyclonedds.core import Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.qos import Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration
from torchvision.models import resnet18
from lerobot.policies.diffusion.modeling_diffusion import _replace_submodules
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import PreTrainedConfig
from lerobot.processor.core import TransitionKey
from sensor_comm_dds.utils.liveliness_listener import LivelinessListener
from sensor_comm_dds.communication.data_classes.clotheshanger import Clotheshanger
import numpy as np
import time
import torch
import cv2
from loguru import logger
import rerun as rr
import torchvision.transforms as transforms
from torch import nn


class ClothesHanger:
    def __init__(self, baseline=np.array([None, None, None, None])):
        listener = LivelinessListener(topic_name="Clotheshanger")
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, topic_name="Clotheshanger", data_type=Clotheshanger)
        qos = Qos(Policy.History.KeepLast(1))  # Only ever take the last available data published to a topic

        self.baseline = baseline
        self.offsets = np.array([0, 0, 0, 0])
        self.thresholds = np.array([None for _ in range(4)])
        self.reader = DataReader(domain_participant, topic, listener=listener, qos=qos)

    def read(self):
        values_raw = np.array(self.reader.take_one(timeout=duration(seconds=5)).taxel_values).astype(np.uint8) - self.offsets
        if self.baseline.all():
            values = np.maximum(self.baseline - values_raw, 0)
        else:
            values = values_raw.copy()
        if self.thresholds.all():
            values = (values > self.thresholds)*self.baseline
        return values, values_raw
    
    def init_thresholds(self, threshold=20):
        if self.baseline.all():
            self.thresholds = np.array([threshold for _ in range(4)])
        else:
            raise ValueError("Baseline values must be provided to initialize thresholds.")
    
    def init_offsets(self, init_vals):
        if not self.baseline.all():
            raise ValueError("Baseline values must be provided to initialize offsets.")
        self.offsets = init_vals - self.baseline

    def init_baseline(self, baseline):
        self.baseline = np.array(baseline)
    
class ClothesHangerMock:
    def read(self):
        return np.array([0 for _ in range(4)]), np.array([0 for _ in range(4)])
    
    def init_offsets(self, _):
        pass
    
    def init_thesholds(self):
        pass

    def init_baseline(self, _):
        pass

class ClothesHangerSpoof:
    def __init__(self):
        bb_checkpoint_path = "outputs/vision_backbones/b-n200-PREPR-INSTR1-all-data/bb_b-n200-PREPR-INSTR1-all-data-epoch15.pth"
        #bb_checkpoint_path = "outputs/vision_backbones/bb_AUG_b-n200-PREPR-INSTR1-fold1-epoch89.pth"
        normaliser_checkpoint_path = "/home/rproesma/Documents/Projects/robot_imitation_glue/outputs/train/b-n200-INSTR1-300k-1enc"  # use pre- and postprocessors trained on INSTR1 dataset: these have stats calculated for clothes hanger values in the state
        ckpt = torch.load(bb_checkpoint_path, map_location="cpu")
        self.model = resnet18(weights=None)
        self.model = _replace_submodules(
			root_module=self.model,
			predicate=lambda x: isinstance(x, nn.BatchNorm2d),
			func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
		)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to("cuda")

        normaliser_config = PreTrainedConfig.from_pretrained(normaliser_checkpoint_path)
        preprocessor, postprocessor = make_pre_post_processors(normaliser_config, pretrained_path=normaliser_checkpoint_path)

        self._preprocessor_normalizer_step = next(
            step for step in preprocessor.steps if step.__class__.__name__ == "NormalizerProcessorStep"
        )
        self._postprocessor_unnormalizer_step = next(
            step for step in postprocessor.steps if step.__class__.__name__ == "UnnormalizerProcessorStep"
        )

    def img_preprocessor(self, img, img_type, crop_width_range=(320, 960),
                        crop_height_range=(0, 720)):
        try:
            crop_height = crop_height_range[1] - crop_height_range[0]
            crop_width = crop_width_range[1] - crop_width_range[0]
            resize = (2*crop_width//3, 2*crop_height//3)
            if img_type == "scene_image":
                processed_img = cv2.resize(np.array(img)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]+70:crop_width_range[1]+70], resize, interpolation=cv2.INTER_LINEAR)
            elif img_type == "wrist_wilson_image":
                processed_img = cv2.resize(np.flip(np.array(img), axis=0)[crop_height_range[0]:crop_height_range[1], crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
            else:
                raise ValueError(f"Unknown image type: {img_type}")
            processed_img = torch.tensor(processed_img).float() / 255.0
            rr.log(f"obs_prepr: {img_type}", rr.Image(processed_img))
            return processed_img.permute(2, 0, 1).unsqueeze(0)
        
        except Exception as e:
            logger.error(f"Error processing image {img}, {img.shape}: {e}")
            raise e

    def read(self, scene_img, wrist_img):
        t_start = time.time()
        scene_img = self._normalize_image("observation.images.scene_image", self.img_preprocessor(scene_img, "scene_image")).to("cuda")
        wrist_img = self._normalize_image("observation.images.wrist_wilson_image", self.img_preprocessor(wrist_img, "wrist_wilson_image")).to("cuda")

        with torch.no_grad():
            try:
                scene_output = self.model(scene_img).squeeze(0)
                wrist_output = self.model(wrist_img).squeeze(0)
            except Exception as e:
                logger.error(f"Error occurred while processing images [{scene_img.shape}, {wrist_img.shape}]: {e}")
                raise e

        # average two predictions and unnormalize
        prediction = (scene_output + wrist_output) / 2.0
        output = self._unnormalize_predicted_ch_values(prediction).cpu().numpy()
        t_end = time.time()
        logger.info(f"Clothes hanger spoof read ({output}) took {(t_end - t_start)*1000:.2f} ms")
        return output.astype(np.float32)  #np.array([0, 0, 0, 0]).astype(np.float32)
    
    def _normalize_image(self, image_key: str, image: torch.Tensor) -> torch.Tensor:
        transition = {TransitionKey.OBSERVATION.value: {image_key: image}}
        out = self._preprocessor_normalizer_step(transition)
        return out[TransitionKey.OBSERVATION.value][image_key]


    def _unnormalize_predicted_ch_values(self, prediction) -> torch.Tensor:
        post_stats = self._postprocessor_unnormalizer_step.stats

        min_vals = torch.as_tensor(post_stats["observation.state"]["min"], device="cuda")[7:]
        max_vals = torch.as_tensor(post_stats["observation.state"]["max"], device="cuda")[7:]
        denom = max_vals - min_vals
        denom = torch.where(denom == 0, torch.full_like(denom, 1e-8), denom)
        # Inverse of MIN_MAX normalization from [-1, 1] back to [min, max].
        return (prediction + 1.0) * denom / 2.0 + min_vals


if __name__ == "__main__":
    clothes_hanger = ClothesHanger()
    ctr = 0
    t_start = time.time()
    while True:
        print(clothes_hanger.read())
        ctr += 1
        if ctr > 100:
            t_end = time.time()
            print(f'Readout frequency: {ctr / (t_end - t_start)} Hz')
            ctr = 0
            t_start = time.time()