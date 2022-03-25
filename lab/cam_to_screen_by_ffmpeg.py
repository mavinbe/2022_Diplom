import cv2
import os
import atexit

RTSP_URL = 'rtsp://malte:diplom@192.168.0.105:554//h264Preview_07_main'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame = cap.read()
    cv2.imshow('RTSP stream', frame)
    print(".")

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()