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


class ClothesHanger:
    def __init__(self):
        listener = LivelinessListener(topic_name="Clotheshanger")
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, topic_name="Clotheshanger", data_type=Clotheshanger)
        qos = Qos(Policy.History.KeepLast(1))  # Only ever take the last available data published to a topic

        self.offsets = np.array([0, 0, 0, 0])
        self.reader = DataReader(domain_participant, topic, listener=listener, qos=qos)

    def read(self):
        values = np.array(self.reader.take_one(timeout=duration(seconds=5)).taxel_values).astype(np.uint8) - self.offsets
        return values
    
    def set_offsets(self, offsets):
        pass
    
class ClothesHangerMock:
    def read(self):
        return np.array([0 for _ in range(4)])
    
    def set_offsets(self, offsets):
        pass


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