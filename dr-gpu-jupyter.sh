#!/usr/bin/env bash

if [[ -z "${DATA_PATH}" ]]; then
    export DATA_PATH=/data
fi

docker run --rm -it\
 --runtime=nvidia\
 -p 8888:8888\
 -p 6006:6006\
 -v ${DATA_PATH}:/data\
 -v ${PWD}:/notebooks\
 -w /notebooks\
 mask_rcnn/mask_rcnn\
 jupyter lab --ip=0.0.0.0 --NotebookApp.token='' --allow-root --no-browser
