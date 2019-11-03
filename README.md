# MTCNN

`pytorch` implementation of **inference stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

## Example
![example of a face detection](images/example.png)

## How to use it
Install the package using pip:
```bash
pip install mtcnn-pytorch
```

Example usage:
```python
from mtcnn import detect_faces
import cv2

image = cv2.imread('image.jpg')
bounding_boxes, landmarks = detect_faces(image)
```
For examples see `test_on_images.ipynb`.

## Requirements
* pytorch 1.0
* opencv-python, numpy

* Pillow for visualization_utils

## Credit
This implementation is heavily inspired by:
* [pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)  