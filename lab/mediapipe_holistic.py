import time

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

# For webcam input:
cap = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output014.mp4")
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  print('----INFERENCE TIME----')
  while True:

    start = time.perf_counter()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = holistic.process(image)


    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    landmark_list = results.face_landmarks
    #print(type(landmark_list))
    if type(landmark_list) is landmark_pb2.NormalizedLandmarkList:
        landmark_list = list(enumerate(landmark_list.landmark))
        landmark_list = list(filter(lambda x: x[0] in [263,7], landmark_list))
        print(len(landmark_list))
        if len(landmark_list) >= 1:
            _, left_eye = landmark_list[1]
            _, right_eye = landmark_list[0]
            print(str(left_eye.x - right_eye.x) + " " + str(left_eye.x) + " " + str(right_eye.x))
        # exit()

    # mp_drawing.draw_landmarks(
    #     image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_contours_style())
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp_holistic.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles
    #     .get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    inference_time = time.perf_counter() - start
    #print('%.2f ms' % (inference_time * 1000))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
