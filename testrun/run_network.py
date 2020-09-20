
import os
import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stat import ST_CTIME, ST_MODE, S_ISREG
import time
import json
from utils import get_model_by_type

class RunNetworkClass():
    def __init__(self):
        print("[RunNetworkClass] start constructor")
        self.model_name_to_use = self.get_initial_model_name()
        print("[RunNetworkClass] Model name to use: {}".format( self.model_name_to_use))
        self.kl = get_model_by_type()
        self.load_model(self.kl, self.model_name_to_use)

    def run(self, image):
        outputs = self.kl.run(image)
        print("[RunNetworkClass] Output network : steering={} throttle={}".format(outputs[0],outputs[1]))
        return outputs

    def get_initial_model_name(self, dirpath='../models'):
        entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
        entries = ((os.stat(path), path) for path in entries)

        # leave only regular files, insert creation date
        entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]) and path.endswith(".h5"))
        list_from_entries = list(sorted(entries))
        print("[RunNetworkClass] Current list with models : {}".format(list_from_entries))
        print("{}".format(len(list_from_entries)))
        if len(list_from_entries) > 0:
            return list_from_entries[-1][1]
        return "bla"

    def use_new_model(self, model_name):
        print("[RunNetworkClass] Use new model '{}'".format(model_name))
        self.model_name_to_use = "/data/{}".format(model_name)
        self.kl = get_model_by_type()
        self.load_model(self.kl, self.model_name_to_use)

        print(("[RunNetworkClass] Model loaded".format()))

    def load_model(self, kl, model_path):
        start = time.time()
        try:
            print('[RunNetworkClass] loading model {}'.format(model_path))
            kl.load(model_path)
            print('[RunNetworkClass] finished loading in {} sec.'.format(str(time.time() - start)))
        except Exception as e:
            print(e)
            print('ERR>> problems loading model', model_path)


def draw_curve(image, speed, rotate, color):
    points = [[80, 120]]
    numpoints = 5
    for x in range(1, numpoints + 1):
        points.append([80 + rotate * 30 * (x / numpoints) ** 2, 120 + (speed * 60 * x / numpoints)])
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], False, color)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    os._exit(0)

def ConvertToUseful(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 50) % 179

    target = np.uint8([[[255, 0, 0]]])
    targethsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)
    targeth = (targethsv[0][0][0] + 50) % 179
    # print(targeth, targethsv)
    lower = np.array([targeth-10, 120, 25])
    upper = np.array([targeth+10, 255, 225])
    mask = cv2.inRange(hsv, lower, upper)
    mask3 = np.zeros_like(img)
    for i in range(3):
        mask3[:, :, i] = mask[:, :]
    return mask3

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    network = RunNetworkClass()

    dir = "../data/tubs/tub_6/"
#    files = os.listdir(dir)
    entries = (os.path.join(dir, fn) for fn in os.listdir(dir))
    entries = ((os.stat(path), path) for path in entries)
    entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]) and path.endswith(".json"))
    files = list(sorted(entries))

    for file in files:
        if file[1].endswith(".json"):
            print(file)
            with open(file[1]) as json_file:
                data = json.load(json_file)
                print(data["cam/image_array"])

                cv_image = cv2.imread(".." + data["cam/image_array"])
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                cv_image = cv2.resize(cv_image, (160, 120))
                cv_image = ConvertToUseful(cv_image)
                outputs = network.run(cv_image)

                steering = outputs[0]
                throttle = outputs[1]

                print("{} <> {}  |  {} <> {}".format(throttle, data["user/throttle"], steering, data["user/angle"]))

                draw_curve(cv_image, data["user/throttle"], data["user/angle"], (0, 255, 0))
                draw_curve(cv_image, throttle, steering, (0, 255, 255))
                plt.imshow(cv_image)
                plt.show(block=False)
                plt.pause(0.001)

    plt.show()


