import os
import sys
os.chdir('.')
# Src directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

import socket
import time
from imutils.video import VideoStream
import imagezmq as imagezmq
import cv2

class ImageSendingProvider:
    def __init__(self, server_ip='*', server_port=None):
        if server_port is None:
            print("Error: no server_port are given.")
            exit(1)
        print("ImageSendingProvider: connect_to : "+'tcp://'+str(server_ip)+':'+str(server_port))
        self.sender = imagezmq.ImageSender(connect_to='tcp://'+str(server_ip)+':'+str(server_port), REQ_REP=False)
        self.hostname = socket.gethostname()

    def send(self, img=None):
        if img is None:
            print("Error: no img are given.")
            exit(1)
        jpeg_quality = 95  # 0 to 100, higher is better quality, 95 is cv2 default
        _, jpg_buffer = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        #self.sender.send_jpg(self.hostname, jpg_buffer)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print(img.shape)
        self.sender.send_image(self.hostname, img)