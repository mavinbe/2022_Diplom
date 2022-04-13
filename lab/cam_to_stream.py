# send to stream_to_screen.py

import cv2
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../.."))

cap  = cv2.VideoCapture(ROOT_DIR+"/data/05_20211102141647/output014.mp4")
#cap  = cv2.VideoCapture('videotestsrc is-live=true ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

framerate = 25
width = int(1280)
height = int(960)

#out = cv2.VideoWriter('appsrc ! videoconvert ! videorate ! video/x-raw,width=1280,height=960,framerate=25/1 ! jpegenc ! rtpjpegpay  ! udpsink host=192.168.178.46  port=7001',
#                      0, framerate, (1280, 960))
out = cv2.VideoWriter("appsrc ! videoconvert ! video/x-raw,format=I420 ! jpegenc ! rtpjpegpay ! rtpstreampay ! udpsink host=192.168.178.46 port=7001", cv2.CAP_GSTREAMER, 0, framerate, (width, height), True)


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #frame = frame[478:, :]
        frame = cv2.resize(frame, (width, height))
        #print(frame.shape)

        #cv2.imshow("asd", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
