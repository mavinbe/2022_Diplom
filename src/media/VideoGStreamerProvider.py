# file: videocaptureasync.py
import threading
import os
import cv2
import time

from datetime import datetime

class VideoStreamProvider:
    def __init__(self, src=None, width=None, height=None, play_back_speed=1):
        if src is None:
            print('Error: stream path for VideoStreamProvider not set.')
            exit()

        print("VideoStreamProvider: load images from " + str(src))

        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.play_back_speed = play_back_speed
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 25.04 * 60 * 0)

        if width is None and height is None:
            print('Use native width & height')
        else:
            print('Use given width & height: '+str(width)+" "+str(height))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps is not None:
            print("Frames per second using self.cap.get(cv2.CAP_PROP_FPS) : {0}".format(self.fps))
        else:
            self.fps = 5
            print("Frames per second using default : {0}".format(self.fps))


        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.update_count = 0
        self.read_count = 0

        self.start()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def get(self, var1):
        self.cap.get(var1)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        frame_time = 1/self.fps / self.play_back_speed
        start_time = time.time()
        last_time = start_time
        print ("frame_time "+str(frame_time))
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.update_count += 1

                diff = time.time() - last_time
                
                # print(time.time() -last_time)
                #print(self.update_count / (time.time() - start_time))
                last_time = time.time()
            time.sleep(max(0, frame_time - diff))


    def read(self):
        frame = None
        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()
            else:
                print("frame was not grabbed "+str(self.read_count))
            self.read_count += 1
        # convert to RGB
        return self.grabbed, frame

    def release(self):
        self.started = False
        self.thread.join()
        self.cap.release()
        print("Cam released")


    def __exit__(self, exec_type, exc_value, traceback):
        self.release()


