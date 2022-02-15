# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""

import argparse
import time

import numpy
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import os

from matplotlib import cm

_NUM_KEYPOINTS = 17

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_COLORS = {
    0: (255, 0, 0),
    1: (255, 255, 0),
    2: (255, 255, 0),
    3: (255, 0, 0),
    4: (255, 0, 0),
    5: (255, 0, 0),
    6: (255, 0, 0),
    7: (255, 0, 0),
    8: (255, 0, 0),
    9: (255, 0, 0),
    10: (255, 0, 0),
    11: (0, 255, 255),
    12: (0, 255, 255),
    13: (255, 0, 0),
    14: (255, 0, 0),
    15: (255, 0, 0),
    16: (255, 0, 0)
}

file_path = os.path.dirname(__file__)


def main(img=None, frameId=0, detectionId=0):
    print(numpy.shape(img))
    img = Image.fromarray(img)
    interpreter = get_interpreter()
    if img is None:
        img = get_image()
    resized_img = get_resized_img(img, interpreter)
    pose = get_pose(interpreter, resized_img)
    width, height = img.size
    pose_in_original_coordinates = transform_pose_to_orignal_coordinates(pose, width, height)
    serialized_pose = serialize_pose_list(pose_in_original_coordinates)
    write_to_result_file(serialized_pose)
    draw_pose_and_save_img(img, pose, frameId, detectionId)
    return pose


def write_to_result_file(serialized_pose):
    f = open(file_path + "/result.txt", "w")
    f.write(serialized_pose)
    f.close()


def serialize_pose_list(pose):
    array_1_dim = pose.reshape(-1)
    string = ' '.join(str(e) for e in array_1_dim)
    return string


def deserialize_pose_list(string):
    array_1_dim_recover = numpy.asarray(string.split(' '), numpy.float64)
    matrix_recover = array_1_dim_recover.reshape((-1, 17, 3))
    return matrix_recover


def transform_pose_to_orignal_coordinates(pose, width, height):
    pose_clone = pose.copy()
    for i in range(0, _NUM_KEYPOINTS):
        #print(f"Hello, {pose_clone[i][1]}. You are {pose[i][1]} {width}.")
        pose_clone[i][1] = round(pose[i][1] * width, 0)
        pose_clone[i][0] = round(pose[i][0] * height)
    return pose_clone


def draw_pose_and_save_img(img, pose, frameId, detectionId):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for i in range(0, _NUM_KEYPOINTS):
        if pose[i][2] >= 0.2:
            draw.ellipse(
                xy=[
                    pose[i][1] * width - 2, pose[i][0] * height - 2,
                    pose[i][1] * width + 2, pose[i][0] * height + 2
                ],
                fill=KEYPOINT_COLORS[i])
    img.save(file_path + f"/result_{frameId}_{detectionId}.bmp")


def get_pose(interpreter, resized_img):
    common.set_input(interpreter, resized_img)
    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(5):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
        print('%.2f ms' % (inference_time * 1000))

    return pose


def get_resized_img(img, interpreter):
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
    return resized_img


def get_interpreter():
    interpreter = make_interpreter(file_path + "/movenet_single_pose_lightning_ptq_edgetpu.tflite")
    interpreter.allocate_tensors()
    return interpreter


def get_image():
    img = Image.open(file_path + "/squat.bmp")
    print(type(img))
    return img


if __name__ == '__main__':
    print('MAIN')
    main()
