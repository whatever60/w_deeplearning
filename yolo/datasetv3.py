import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Dataloader

from PIL import Image, ImageFile
from pytorch_lightning.metrics.functional import iou


class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size, S, C, transform):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thres = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=' ', ndmin=2), 4, axis=1).tolist()
        img_path =  os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            # calculate iou for a particular box and all the anchors
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchros_per_scale  # 0, 1, 2, which scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # 0, 1, 2, which anchor in that scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thres:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return image, tuple(targets)
