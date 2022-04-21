# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from dataset.yolo5_dataset import LoadImages
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class ObjectTracker:
    def __init__(self, yolo_model="Yolov5_DeepSort_Pytorch/yolov5/weights/crowdhuman_yolov5m.pt", deep_sort_model = "osnet_x0_25", imgsz=[640, 640], half=False, show_vid=False):
        with torch.no_grad():
            device = ""
            dnn = True
            config_deepsort = "Yolov5_DeepSort_Pytorch/deep_sort/configs/deep_sort.yaml"
            self.augment = False
            self.conf_thres = 0.3
            self.iou_thres = 0.5
            self.classes = [0]
            self.agnostic_nms = False
            self.max_det = 1000

            device = select_device(device)
            # initialize deepsort
            cfg = get_config()
            cfg.merge_from_file(config_deepsort)
            self.deepsort = DeepSort(deep_sort_model,
                                device,
                                max_dist=cfg.DEEPSORT.MAX_DIST,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                _lambda=0
                                )

            # Initialize

            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
            stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Half
            half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
            if pt:
                model.model.half() if half else model.model.float()

            # Check if environment supports image displays
            if show_vid:
                show_vid = check_imshow()

            # Dataloader
            cudnn.benchmark = True  # set True to speed up constant image size inference

            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup


            self.dataset = LoadImages(img_size=imgsz, stride=stride, auto=pt and not jit)
            self.device = device
            self.half = half
            self.model = model
            self.show_vid = show_vid


    def inference_frame(self, im0s):
        with torch.no_grad():
            img = self.dataset.prepare_image_for_model(im0s)
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            # Inference
            pred = self.model(img, augment=self.augment, visualize=False)
            t3 = time_sync()
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            result_dict = {}
            t4, t5, t6 = t3, t3, t3
            confirmed_id_list = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                im0 = im0s.copy()
                annotator = Annotator(im0, line_width=2, pil=not ascii)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class


                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs, confirmed_id_list = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            result_dict[id] = bboxes

                            c = int(cls)  # integer class
                            label = f'{id}  {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                    t6 = time_sync()

                else:
                    self.deepsort.increment_ages()
                    confirmed_id_list = [(track.track_id, track.state) for track in self.deepsort.tracker.tracks]
                    #LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()
                if self.show_vid:
                    cv2.imshow(str("video"), im0)
                    #if cv2.waitKey(1) == ord('q'):  # q to quit
                    #    pass
                        #raise StopIteration

            t7 = time_sync()

            #LOGGER.info(f'inference_frame:({(t7 - t1) * 1000:.3f}ms) prepare::({(t2 - t1) * 1000:.3f}ms), YOLO::({(t3 - t2) * 1000:.3f}ms), diverses::({(t4 - t3) * 1000:.3f}ms), DeepSort::({(t5 - t4) * 1000:.3f}ms), draw::({(t6 - t5) * 1000:.3f}ms), display::({(t7 - t6) * 1000:.3f}ms)')

            return result_dict, confirmed_id_list


if __name__ == '__main__':

    with torch.no_grad():
        object_tracker = ObjectTracker()

        cap = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output015.mp4")
        while cap.isOpened():
            t1 = time_sync()
            success, im0 = cap.read()
            t2 = time_sync()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue


            t3 = time_sync()
            result_list = object_tracker.inference_frame(im0)
            t4 = time_sync()
            print(result_list)
            #LOGGER.info(f'DONE on prepare:({(t4 - t1)*1000:.3f}ms)    read:({(t2 - t1)*1000:.3f}ms), prepare:({(t3 - t2)*1000:.3f}ms), inference:({(t4 - t3)*1000:.3f}ms)')

    cap.release()