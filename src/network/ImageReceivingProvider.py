import os
import sys
os.chdir('.')
# Src directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from util.Config import Config
import imagezmq as imagezmq
#import simplejpeg
import PIL.Image as Image

class ImageReceivingProvider:
    def __init__(self, server_ip=None, server_port=None):
        def __init__(self, server_ip=None, server_port=None):
            if server_ip is None:
                print("Error: no server_ip are given.")
                exit(1)
            if server_port is None:
                print("Error: no server_port are given.")
                exit(1)

        print("ImageReceivingProvider: open_port at: " + 'tcp://' + str(server_ip) + ':' + str(server_port))
        self.receiver = imagezmq.ImageHub(open_port='tcp://' + str(server_ip) + ':' + str(server_port), REQ_REP=False)

    def read(self):
        sender_name, image = self.receiver.recv_image()

        return sender_name, image
        
        
        sender_name, image = self.receiver.recv_jpg()
        image = Image.frombuffer(image)
        return sender_name, image
 