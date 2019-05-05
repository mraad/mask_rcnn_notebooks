# Mask_RCNN Notebooks (WORK IN PROGRESS)

```bash
conda remove --yes --quiet --all --name mask_rcnn
conda create --yes --quiet --name mask_rcnn --clone arcgispro-py3
conda activate mask_rcnn
conda install --yes --quiet pip python=3.6
python -m pip install --upgrade pip
cd ${MRCNN_HOME}
pip install -r requirements.txt
pip install pycocotools
```
