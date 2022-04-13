# receive from cam_to_stream.py


#import cv2

# gst-launch-1.0 -v playbin uri=rtsp://malte:diplom@192.168.0.105:554//h264Preview_07_main uridecodebin0::source::latency=300
# works on PC

# gst-launch-1.0 rtspsrc location=rtsp://malte:diplom@192.168.0.105:554//h264Preview_07_main latency=100 ! queue ! rtph264depay ! avdec_h264 ! videoconvert ! videoscale  ! autovideosink


#cap_send = cv2.VideoCapture('videotestsrc ! video/x-raw,framerate=20/1 ! videoscale ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
#print(cv2.asd)



# https://stackoverflow.com/questions/66019614/how-to-send-receive-videotestsrc-images-encoded-as-jpeg-using-rtp-with-gstreamer
#
# send command
# gst-launch-1.0 videotestsrc ! videoconvert ! video/x-raw, format=YUY2 ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5000
#
# receive command
# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, framerate=30/1, payload=26, clock-rate=90000 ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink
import time

import cv2
import screeninfo

import atexit
print("asd")

cap = cv2.VideoCapture('udpsrc port=7001 !  application/x-rtp-stream,encoding-name=JPEG ! rtpstreamdepay ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

atexit.register(cap.release)
screen = screeninfo.get_monitors()[0]

print(f"screen {screen}")
#cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
#cv2.moveWindow("frame", int(screen.x - 1), int(screen.y - 1))
#cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print((ret, frame))
    if frame is None:
        time.sleep(1)
        continue

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = frame[:-180, 60:-60]
    #print(frame)
    #print((screen.width, screen.height))
    #frame = cv2.resize(frame, (screen.width, screen.height))
    #print(frame.shape)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # print(f"frame.shape {frame.shape}")
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
