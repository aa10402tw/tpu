# Setup steps on Windows #

## 1. Install package
```
pip install tensorflow-gpu==1.15  # GPU
pip install pytk
pip install --user Cython matplotlib opencv-python-headless pyyaml Pillow
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

## 2. Download Checkpoint 
Download checkpoints for semantic segmantation form [here](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/self_training)

Then save it to ./models/official/detection/projects/self_training/weights/

## 3. Running Models
=== Windows Command (at /tpu)===
```
set PYTHONPATH=%PYTHONPATH%;./models
python ./models/official/detection/inference.py ^
  --model="segmentation" ^
  --image_size=640 ^
  --config_file="./models/official/detection/projects/self_training/configs/pascal_seg_efficientnet-l2-nasfpn.yaml" ^
  --checkpoint_path="./models/official/detection/projects/self_training/weights/efficientnet-l2-nasfpn-ssl/model.ckpt" ^
  --label_map_file="./models/official/detection/datasets/coco_label_map.csv" ^
  --image_file_pattern="./_input/*.jpg" ^
  --output_dir="./_output/"
```
