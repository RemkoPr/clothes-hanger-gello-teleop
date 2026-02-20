from cyclonedds.core import Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.qos import Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration
from sensor_comm_dds.utils.liveliness_listener import LivelinessListener
from sensor_comm_dds.communication.data_classes.clotheshanger import Clotheshanger
import numpy as np
import time
import torch
from soft_sensor.spatial_concat_resnet import SpatialConcatResNet
import cv2
from loguru import logger
import rerun as rr
import torchvision.transforms as transforms


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
        values = np.array(self.reader.take_one(timeout=duration(seconds=5)).taxel_values).astype(np.uint8) - self.offsets
        if self.baseline.all():
            values = np.maximum(self.baseline - values, 0)
        #if self.thresholds.all():
        #    values = (values > self.thresholds)*self.baseline
        return values
    
    def init_thesholds(self):
        if self.baseline.all():
            self.thresholds = self.baseline - 30
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
        return np.array([0 for _ in range(4)])
    
    def init_offsets(self, _):
        pass
    
    def init_thesholds(self):
        pass

    def init_baseline(self, _):
        pass

class ClothesHangerSpoof:
    def __init__(self, baseline=np.array([None, None, None, None]), concat_type="WIDTH"):
        self.model = SpatialConcatResNet(pretrained=False).to("cuda")
        self.model.load_state_dict(torch.load("/home/rproesma/Documents/Projects/ClothesHangUR/Python/soft_sensor/predict_binary/outputs/2025-09-22/spatial_concat_resnet_1.pth", map_location="cuda"))
        self.model.eval()
        self.concat_type = concat_type
        self.baseline = np.array(baseline)
        self.normalise = transforms.Normalize(mean=[0.3365, 0.3911, 0.4145], std=[0.2832, 0.2440, 0.2536])

    def read(self, scene_img, wrist_img):
        t_start = time.time()
        scene_img = self.preprocessor(scene_img)
        wrist_img = self.preprocessor(wrist_img)

        if self.concat_type == "NONE":
            raise NotImplementedError("No concat type not implemented for spoof sensor.")
        elif self.concat_type == "CHANNEL":
            img = torch.cat((scene_img, wrist_img), dim=0)
        elif self.concat_type == "HEIGHT":
            img = torch.cat((scene_img, wrist_img), dim=1)
        elif self.concat_type == "WIDTH":
            img = torch.cat((scene_img, wrist_img), dim=2)
        #rr.log("image_for_spoofch", rr.Image(img))
        logger.debug(f"spoof image first row: {img[0,0,:20]}")
        img = img.float().to("cuda").unsqueeze(0)  # add batch dim
        with torch.no_grad():
            output = self.model(img)
            prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # remove batch dim
            binary_prediction = (prob > 0.5)*self.baseline
        t_end = time.time()
        logger.info(f"Clothes hanger spoof read ({binary_prediction}) took {(t_end - t_start)*1000:.2f} ms")
        return binary_prediction.astype(np.float32)
    
    def preprocessor(self, img):
        crop_width_range = (320+70, 960-70)
        crop_width = crop_width_range[1] - crop_width_range[0]
        crop_height_range = (0, 720)
        crop_height = crop_height_range[1] - crop_height_range[0]
        crop_width = crop_width_range[1] - crop_width_range[0]
        resize = (crop_width, crop_height)
        train_crop_cutoff = (0, 0)
        
        img = cv2.resize(np.array(img)[:, crop_width_range[0]:crop_width_range[1]], resize, interpolation=cv2.INTER_LINEAR)
        img = np.array(img)[train_crop_cutoff[0]:-train_crop_cutoff[0] if train_crop_cutoff[0] > 0 else None, train_crop_cutoff[1]:-train_crop_cutoff[1] if train_crop_cutoff[1] > 0 else None]
        img = torch.tensor(img).permute(2, 0, 1).float()  / 255.0  # to C,H,W
        img = self.normalise(img)

        return img


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