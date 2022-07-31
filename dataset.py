# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
from matplotlib.pyplot import imshow
import cv2 as cv

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self,csv_file, img_path, label_path, S=7, B=2, C=20, transform = None,
    ):
            self.annotations = pd.read_csv(csv_file)
            self.img_path = img_path
            self.label_path = label_path
            self.transform = transform
            self.S = S
            self.B = B
            self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_path, self.annotations.iloc[index,1])
        target_boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                target_boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_path, self.annotations.iloc[index,0])
        image = imshow(img_path)
        target_boxes = torch.tensor(target_boxes)

        if self.transform:
            images, target_boxes = self.transform(image, target_boxes)

        label_matrix = torch.zeros((self.S,self.S,self.C+5 * self.B))

# draw bboxes in image, class_label, x, y, width, height
        for box in target_boxes:
            class_label,x,y,width,height = box.tolist()
            class_label = int(class_label)
            # cell row and cell column in an image of our target (y and x is ratio in an image)
            # int() calculating the start coordinate in cell
            # if S=7, target ratio of x = 0.5 (j will be 3, from cell 3)
            i, j = int(self.S * y), int(self.S * x)
            # the coordinates in that particular cell #0.5 from above example
            x_cell, y_cell = self.S * x-j, self.S * y-j
            # since w, h are reprecenting the whole image from the camera
            width_cell , height_cell = (width * self.S, height * self.S)
            # SxS in an image, 20 classes since for box in target_boxes
            if label_matrix[i,j,20] == 0:
                label_matrix[i,j,20] = 1
                label_matrix[i, j, class_label] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell])
                label_matrix[i,j,21:25] = box_coordinates
                # There is a class in that cell

        return image, label_matrix
