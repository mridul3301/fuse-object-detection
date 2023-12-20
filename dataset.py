import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from utils import * 

# allows PIL to load images even if they are truncated or incomplete
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
  def __init__(self, csv_file, img_dir, label_dir, anchors,
               image_size=416, S=[13,26,52], C=20, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.label_dir = label_dir
    self.transform = transform
    self.S = S

    # Suppose, anchors[0] = [a,b,c], anchors[1] = [d,e,f], anchors[2] = [g,h,i] : Each set of anchors for each scale
    # List addition gives shape 3x3
    # Anchors per scale suggests that there are three different aspect ratios for each anchor position.
    self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # For all 3 scales
    self.num_anchors = self.anchors.shape[0]
    self.num_anchors_per_scale = self.num_anchors // 3

    self.C = C

    # If a cell has obj. then one anchor is responsible for outputting it,
    # one that's responsible is the one that has highest iou with ground truth box
    # but, there might be cases where there are several boxes in the same cell
    self.ignore_iou_thresh = 0.5

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # np.roll with shift 4 on axis 1: [class, x, y, w, h] --> [x, y, w, h, class]

    img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
    image = Image.open(img_path)

    if self.transform:
      image = self.transform(image)

    targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 because objectness score, bounding box coordinates (x, y, w, h), class label

    for box in bboxes:
      """For each box in bboxes,
      we want to assign which anchor should be responsible and
      which cell should be responsible for all the three different scales prediction"""
      iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) # IOU from height and width
      anchor_indices = iou_anchors.argsort(descending=True, dim=0) # Sorting sucht that the first is the best anchor

      x, y, width, height, class_label = box
      has_anchor = [False, False, False] # Make sure there is an anchor for each of three scales for each bounding box

      for anchor_idx in anchor_indices:
        scale_idx = anchor_idx // self.num_anchors_per_scale # scale_idx is either 0,1,2: 0-->13x13, 1:-->26x26, 2:-->52x52
        anchor_on_scale = anchor_idx % self.num_anchors_per_scale # In each scale, choosing the anchor thats either 0,1,2

        S = self.S[scale_idx]
        i, j = int(S*y), int(S*x) # x=0.5, S=13 --> int(6.5) = 6 | i=y cell, j=x cell
        anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

        if not anchor_taken and not has_anchor[scale_idx]:
          targets[scale_idx][anchor_on_scale, i, j, 0] = 1
          x_cell, y_cell = S*x - j, S*y - i # 6.5 - 6 = 0.5 such that they are between [0,1]
          width_cell, height_cell = (
              width*S, # S=13, width=0.5, 6.5
              height*S
          )

          box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

          targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
          targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
          has_anchor[scale_idx] = True

        # Even if the same grid shares another anchor having iou>ignore_iou_thresh then,
        elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
          targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction

    return image, tuple(targets)