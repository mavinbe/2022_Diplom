This file gives an overal overview.

In Log.md u can find a hopefully consistent and daily work log.


## run

```bash
conda activate ./env
```

```bash
python3 movenet/movenet_pose_estimation.py --model movenet/movenet_single_pose_lightning_ptq_edgetpu.tflite --input movenet/squat.bmp
```

```bash
python Yolov5_DeepSort_Pytorch/track.py --source /home/mavinbe/2021_Diplom/2021_Diplom_Lab/Data/08_20211102141647/output014.mp4 --yolo_model Yolov5_DeepSort_Pytorch/yolov5/weights/crowdhuman_yolov5m.pt --classes 0 --show-vid --save-txt
```
