import cv2
import screeninfo

cap = cv2.VideoCapture("rtsp://malte:diplom@192.168.0.105:554//h264Preview_06_main")
screen = screeninfo.get_monitors()[0]
print(f"screen {screen}")
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("frame", int(screen.x - 1), int(screen.y - 1))
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    print(f"frame.shape {frame.shape}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
