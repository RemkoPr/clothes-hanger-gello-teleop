import rerun as rr
import time
import numpy as np

rr.init("testingringin", spawn=True)

t = np.array([1, 3, 4, 6])
while True:
    time.sleep(1)
    rr.log("curve_test2", rr.Scalars((t%10)**2), rr.SeriesLines(widths=10))
    t += 1