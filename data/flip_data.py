import os
import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stat import ST_CTIME, ST_MODE, S_ISREG
import time
import json

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    os._exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    dirSrc = "/data/tubs/tub_1_20-09-07/"
    dirDst = "/data/tubs/tub_1_20-09-07_flip/"
#    files = os.listdir(dir)
    entries = (os.path.join(".." + dirSrc, fn) for fn in os.listdir(".." + dirSrc))
    entries = ((os.stat(path), path) for path in entries)
    entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]) and path.endswith(".json"))
    files = sorted(list(entries))

    for file in files:
        if file[1].endswith(".json"):
            print(file)
            with open(file[1]) as json_file:
                data = json.load(json_file)
                print(data["cam/image_array"])
                dataPathDst = dirDst + os.path.basename(data["cam/image_array"])
                print(".." + data["cam/image_array"])
                cv_image = cv2.imread(".." + data["cam/image_array"])
                cv_image = cv2.flip(cv_image, 1)
                cv2.imwrite(".." + dataPathDst, cv_image)

                data["cam/image_array"] = dataPathDst
                data["user/angle"] = -data["user/angle"]
                jsonFilenameDst = ".." + dirDst + os.path.basename(file[1])
                with open(jsonFilenameDst, 'w') as outfile:
                    json.dump(data, outfile)
