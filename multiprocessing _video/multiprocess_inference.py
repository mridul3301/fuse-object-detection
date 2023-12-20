import torch
import torch.nn as nn
import numpy as np
import cv2
import importlib
import random
import time

import sys
import os
import multiprocessing

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

from nets.model_main import ModelMain
from mp_utils import YOLOPost, non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####### Load the model -- config , data parallel, restore pretrain model
params_path = "params.py"
config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

model = ModelMain(config, is_training=False)
model.train(False)

# Set data parallel
model = nn.DataParallel(model)
model = model.to(device)

# Restore pretrain model
if config["pretrain_snapshot"]:
    state_dict = torch.load(config["pretrain_snapshot"], map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
else:
    raise Exception("missing pretrain_snapshot!!!")

# print(model)


# YOLO loss with 3 scales
yolo_losses = []
for i in range(3):
    # yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
    #                             config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    yolo_losses.append(YOLOPost(config["yolo"]["anchors"][i],
                                config["yolo"]["classes"], (config["img_w"], config["img_h"])))



####### Inference


def inference(read_frame_queue, batch_detections_queue):
    # time.sleep(2)
    while True:
        if read_frame_queue.qsize() != 0:
            # print(read_frame_queue.qsize())

            frame = read_frame_queue.get()
            # cv2.imshow("frame", frame)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                                        interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

            # print(image.shape) # (3,416,416)

            image = torch.from_numpy(image)
            image = image.unsqueeze(0)

            # print(image.shape) # ([1,3,416,416])


            ####### Inference

            # Perform inference
            with torch.no_grad():
                out = model(image.to(device))
            
            # print(out[0].shape) # ([1,255,13,13]) -> torch

            # Convert tensor to numpy
            output = []
            output.append(out[0].cpu().numpy())
            output.append(out[1].cpu().numpy())
            output.append(out[2].cpu().numpy())
            # print(len(output))
            # print(output[0].shape) # (1, 255, 13, 13) -> numpy

            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i].forward(output[i]))
            output_con = np.concatenate(output_list, 1)


            # print(len(output_list))
            # print(output_list[0].shape) # ([1, 507, 85]) | 13*13*3
            # print(output_list[1].shape) # ([1, 2028, 85]) | 26*26*3
            # print(output_list[2].shape) # ([1, 8112, 85]) | 52*52*3

            # print(len(output_con))
            # print(output_con.shape) # ([1, 10647, 85])

            batch_detections = non_max_suppression(output_con, config["yolo"]["classes"],
                                                            conf_thres=config["confidence_threshold"],
                                                            nms_thres=0.45)

            # print(batch_detections)
            batch_detections_queue.put(batch_detections)
       


def read_video(video_path, read_frame_queue, ori_frame_queue):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")

    # Loop through the frames of the video
    while True:
        ret, frame = cap.read()
        ori_frame = frame

        if not ret:
            break  # Break the loop if there are no more frames

        read_frame_queue.put(frame)
        ori_frame_queue.put(ori_frame)
    cap.release()

    # print(read_frame_queue.qsize())
    # print(ori_frame_queue.qsize())

    # inference(read_frame_queue, ori_frame_queue)


def display(ori_frame_queue, batch_detections_queue):
    # time.sleep(4)
    while True:
        if ori_frame_queue.qsize() != 0 and batch_detections_queue.qsize() != 0:
            ori_frame = ori_frame_queue.get()
            batch_detections = batch_detections_queue.get()   
    
            ######## Plot prediction with bounding box
            classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
            # print(classes)

            for idx, detections in enumerate(batch_detections):
                if detections is not None:
                    # print("Detecting...")

                    # print(im.shape) # eg. (428, 640, 3)
                    unique_labels = np.unique(detections[:, -1])
                    n_cls_preds = len(unique_labels)
                    bbox_colors = {int(cls_pred): (0,255,0) for cls_pred in unique_labels}
                    
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        # print("bbox detect")
                        color = bbox_colors[int(cls_pred)]

                        # Rescale coordinates to original dimensions
                        ori_h, ori_w, _ = ori_frame.shape
                        pre_h, pre_w = config["img_h"], config["img_w"]
                        box_h = ((y2 - y1) / pre_h) * ori_h
                        box_w = ((x2 - x1) / pre_w) * ori_w
                        y1 = (y1 / pre_h) * ori_h
                        x1 = (x1 / pre_w) * ori_w

                        # Create a Rectangle patch
                        cv2.rectangle(ori_frame, (int(x1), int(y1)), (int(x1 + box_w), int(y1 + box_h)), color, 2)

                        # Add label
                        label = classes[int(cls_pred)]
                        cv2.putText(ori_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
                    cv2.imshow('Pred', ori_frame)
                    cv2.waitKey(1)
    cv2.destroyAllWindows()


def main():
    video_path = "./test_video/test102.mp4" # Not live input video
    #video_path = 0 # Live video feed 

    # Create shared queue
    multiprocessing.set_start_method('spawn')
    read_frame_queue = multiprocessing.Queue()
    ori_frame_queue = multiprocessing.Queue()
    batch_detections_queue = multiprocessing.Queue()

    # read_video(video_path, read_frame_queue, ori_frame_queue)

    p1 = multiprocessing.Process(target=read_video, args=(video_path, read_frame_queue, ori_frame_queue))
    p2 = multiprocessing.Process(target=inference, args=(read_frame_queue, batch_detections_queue))
    p3 = multiprocessing.Process(target=display, args=(ori_frame_queue, batch_detections_queue))


    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


if __name__ == "__main__":
    main()


