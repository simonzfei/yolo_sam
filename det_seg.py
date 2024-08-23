# -*- coding: utf-8 -*-
"""
@Time ： 2023
@Auth ： simonzfei
@IDE ：PyCharm
@Motto：thinking coding
"""


import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
import os
import sys
import json
import time
import requests
from requests.adapters import HTTPAdapter
from pathlib import Path
from utils.img_decode import Decode_Img

from segment_anything.segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

reqs = requests.Session()
reqs.mount('http://', HTTPAdapter(max_retries=3))
reqs.mount('https://', HTTPAdapter(max_retries=3))


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def load(model_type, model_name, model_path, *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    print("model_name", model_name, 'model_path', model_path)
    if "yolov7_sq" == model_name:
        model = attempt_load(model_path, map_location=device)  # load FP32 model
        return model
    elif "sam" == model_name:
        sam = sam_model_registry["vit_h"](model_path)
        sam.to(device)
        model = SamPredictor(sam)
        return model

    return None




def call(arg, model, *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    imgsz = 640
    opt_conf_thres = 0.01  # conf score thresh
    opt_iou_thres = 0.5  # iou thresh
    # Initialize
    set_logging()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = check_img_size(imgsz, s=32)  # check img_size
    painting_model = model["painting"]
    if half:
        painting_model.half()  # to FP16
    names = model.module.names if hasattr(painting_model, 'module') else painting_model.names
    # todo  传入 url
    imgage, url, msgs = Decode_Img()(arg)

    t1 = time.time()

    if not isinstance(imgage, np.ndarray):
        print(msgs)
        return {'status': 40309,"errorMsg": f"Failed to get image . please enter correct information {msgs} "}


    h, w, _ = imgage.shape
    img_size = h * w
    im0s = np.copy(imgage)

    img = letterbox(imgage, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    print(img.shape)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = painting_model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    ret = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                conf = round(float(conf), 2)
                xyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                w = (int(xyxy[2]) - int(xyxy[0]))
                h = (int(xyxy[3]) - int(xyxy[1]))
                img_size_target = w * h
                ratio = img_size_target / img_size
                ret.append({"label": label, "socre": conf, "location": xyxy, "rate": ratio})

    t4 = time.time()
    print(ret)
    res_msg = {'status': 0, "data": ret, "inf_time": str(t4 - t0)}
    print(res_msg)

    name = arg["url"].split("/")[-1]
    sam_seg_lable(image=imgage,name=name, data=ret, predictor=model_dict['sam'])

    return res_msg

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    print('mask.shape', mask.shape)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def sam_seg_lable(image,name, data, predictor):
    """
     [
     {'label': 'yellow_pipe', 'socre': 0.92, 'location': [0, 486, 386, 620], 'rate': 0.04209309895833333},
     {'label': 'yellow_pipe', 'socre': 0.93, 'location': [714, 448, 1276, 581], 'rate': 0.060828450520833334},
     {'label': 'red_pipe', 'socre': 0.96, 'location': [374, 0, 741, 598], 'rate': 0.17860188802083332}
     ]}

    """
    current_path = os.getcwd()
    save_path = os.path.join(current_path, str(name).split('.')[0])
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(image.shape)
    h, w, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    save_id = 0
    labelme_data = {
        "version": "4.6.0",
        "flags": {},
        "shapes": [],
        "imagePath": str(name),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    newshape = []
    for obj in data:

        label = obj['label']
        socre = obj['socre']
        location = obj['location']
        input_box = np.array(location)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        print('maskk---------------------', masks)
        mask_img = torch.zeros(masks.shape[-2:])
        mask_img[masks[0] == True] = 255
        save_mask = mask_img.numpy()

        save_name = os.path.join(save_path, str(str(name).split('.')[0] + "_" + str(save_id) + "_" + str(label) + '.jpg'))
        cv2.imwrite(save_name, save_mask)
        save_id += 1

        mask = cv2.imread(save_name, cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a blank image to draw the contours on
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        contour_image = np.zeros_like(mask)

        # Draw the contours on the blank image
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

        # Save the image with contours
        cv2.imwrite(os.path.join(save_path, str('contours'+ '_' + str(save_id) + "_" + str(label) + '.jpg')), contour_image)

        points = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        contour = contours[0]

        clockwise_contour = sort_points_clockwise(contours)
        # Save the contour coordinates to a text file
        i = 0
        for point in clockwise_contour:
            point = point.tolist()
            for po in point:
                if i % 20 == 0:
                    x, y = po[0]
                    points.append([int(x), int(y)])
                i += 1

        shape = {

            "label": str(label),
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        newshape.append(shape)
    labelme_data.update({"shapes": newshape})
    savejson = open( os.path.join(save_path,  str(name).split('.')[0] + '.json'), "w", encoding='utf-8')
    savejson.write(json.dumps(labelme_data, ensure_ascii=False, indent=4))
    savejson.close()


def sort_points_clockwise(points):
    center = tuple(np.mean(points, axis=0))
    return sorted(points, key=lambda point: np.arctan2(point[0] - center[0], point[1] - center[1]))


if __name__ == '__main__':
    painting_model = load('', "yolov7_sq", "/fei/yolo_sam/weights/pipe_painting.pt")
    model_name = 'sam'
    model_path = '/fei/yolo_sam/weights/sam_vit_h_4b8939.pth'
    sam = load("", model_name, model_path)
    model_dict = {"painting": painting_model, "sam": sam}

    arg = {"url": "https://.../100041747820612120412953659717.jpg"}
    result = call(arg, model_dict)
    print(result)