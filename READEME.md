# Active Learning

## How to Install

### with static forward functions

- RTX 3080
- nvidia-driver-545
- CUDA 11.7

```bash
conda create -n py310 python=3.10
conda activate py310

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tqdm matplotlib dataclasses pillow
pip install opencv-python wandb pycocotools

sh data/scripts/COCO2014.sh
```

### original

- RTX 2080
- nvidia-driver-545
- CUDA 10.0

```bash
sudo apt install curl

conda create -n py37 python=3.7
conda activate py37

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install tqdm matplotlib dataclasses pillow=6.1
pip install opencv-python wandb pycocotools

sh data/scripts/COCO2014.sh
```

## Pascal VOC0712 Dataset

### Description
The Pascal VOC0712 dataset is a popular benchmark dataset for object detection and segmentation tasks in computer vision. It consists of images from various real-world scenarios, with objects of interest annotated by bounding boxes or segmentation masks.

The dataset is a combination of two subdirectories, VOC2007 and VOC2012, which together include 20 object classes such as aeroplane, bicycle, car, and person.

### Files
- Annotations folder: Contains XML files that describe the annotations for each image in the dataset.
- ImageSets folder: Contains text files that list the images and their corresponding annotations for the different dataset splits (train, val, test).
- JPEGImages folder: Contains the original JPEG images for the dataset.
- SegmentationClass folder (optional): Contains segmentation maps for some of the images in the dataset.
- SegmentationObject folder (optional): Contains object-level segmentation maps for some of the images in the dataset.

### Data Dictionary
The following table describes the fields in the dataset annotations:

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| ImageID    | string    | Unique identifier for the image |
| FileName   | string    | Filename of the image |
| Width      | integer   | Width of the image in pixels |
| Height     | integer   | Height of the image in pixels |
| ObjectID   | integer   | Unique identifier for the object |
| Name       | string    | Name of the object class |
| XMin       | float     | Minimum X coordinate of the bounding box (normalized between 0 and 1) |
| YMin       | float     | Minimum Y coordinate of the bounding box (normalized between 0 and 1) |
| XMax       | float     | Maximum X coordinate of the bounding box (normalized between 0 and 1) |
| YMax       | float     | Maximum Y coordinate of the bounding box (normalized between 0 and 1) |

Note that the annotations are provided separately for each subdirectory (VOC2007 and VOC2012), but the ImageSets and JPEGImages folders are shared between them.
