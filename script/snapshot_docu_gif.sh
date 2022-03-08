#!/usr/bin/env bash


GIT_HASH=$(git rev-parse --short HEAD)
DATE_TIME=$(date '+%Y-%m-%dT%H:%M:%S');
OUTPUT_FILE="version_control_meta/${DATE_TIME}_${GIT_HASH}.gif"
echo $OUTPUT_FILE

eval "$(conda shell.bash hook)"
conda activate /home/mavinbe/2021_Diplom/2022_Diplom/env/Yolov5_DeepSort_Pytorch
python src/app/run_pipeline_for_docu.py --output-file "$OUTPUT_FILE"
