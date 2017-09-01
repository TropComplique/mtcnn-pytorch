import numpy as np
import torch
from torch.autograd import Variable

import sys
sys.path.append('nets/')
from get_nets import PNet, RNet, ONet
from important_parts import run_first_stage, nms, convert_to_square,\
    calibrate_box, preprocess, get_image_boxes


def detect_faces(img, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8]):
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        img: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes with facial landmarks

    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # BUILD AN IMAGE PYRAMID
    width, height = img.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(img, pnet, s, thresholds[0])
        bounding_boxes.append(boxes)

    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    pick = nms(bounding_boxes[:, 0:5], 0.7, 'union')
    bounding_boxes = bounding_boxes[pick]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, img, size=24)
    img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()
    probs = output[1].data.numpy()

    passed = np.where(probs[:, 1] > thresholds[1])
    bounding_boxes = bounding_boxes[passed]

    bounding_boxes[:, 4] = probs[passed, 1].reshape((-1,))
    offsets = offsets[passed]

    pick = nms(bounding_boxes, 0.7, 'union')
    bounding_boxes = bounding_boxes[pick]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[pick])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, img, size=48)
    img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()
    offsets = output[1].data.numpy()
    probs = output[2].data.numpy()

    passed = np.where(probs[:, 1] > thresholds[2])
    bounding_boxes = bounding_boxes[passed]

    bounding_boxes[:, 4] = probs[passed, 1].reshape((-1,))
    offsets = offsets[passed]
    landmarks = landmarks[passed]

    # compute landmark points
    bbw = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    bbh = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    landmarks[:, 0:5] = np.expand_dims(bounding_boxes[:, 0], 1) +\
        np.expand_dims(bbw, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(bounding_boxes[:, 1], 1) +\
        np.expand_dims(bbh, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    pick = nms(bounding_boxes, 0.7, 'min')
    bounding_boxes = bounding_boxes[pick]
    landmarks = landmarks[pick]

    return bounding_boxes, landmarks
