from os import makedirs

import cv2

import imageio
import argparse
from datetime import datetime

from app.run_pipeline import run, show_image


def display_and_append(image, t):
    show_image(image)
    images.append(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, help='output file')
    opt = parser.parse_args()
    images = []

    try:
        run(display_and_append)
    finally:
        rezied_images = []
        for idx,image in enumerate(images):
            if idx % 4 == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rezied_images.append(cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST))

        imageio.mimsave(opt.output_file, rezied_images)