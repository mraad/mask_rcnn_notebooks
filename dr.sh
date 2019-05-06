#!/usr/bin/env bash

case $1 in
    cpu ) export RUNTIME="";;
    gpu ) export RUNTIME="--runtime=nvidia";;
    * ) echo "Expecting 'cpu' or 'gpu' (no quotes :-) as an argument !"
        exit 1
esac

if [[ -z "${DATA_PATH}" ]]; then
    export DATA_PATH=/data
fi

docker run --rm -it ${RUNTIME}\
 -p 8888:8888\
 -p 6006:6006\
 -v ${DATA_PATH}:/data\
 -v ${PWD}:/notebooks\
 -w /notebooks\
 mask_rcnn/mask_rcnn\
 jupyter notebook --ip=0.0.0.0 --NotebookApp.token='' --allow-root --no-browser
