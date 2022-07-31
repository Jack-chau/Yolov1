import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from functions import (
    intersetion_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss


seed = 123
torch.manual_seed(seed)
# Hyperparameter
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 # paper is 64
weight_decay = 0 # drop_out is 0.5 in paper
eport_num = 100
num_workers = 2 # feed batch_size data to ram, usually set according to the cpu core
pin_memory = True # pinning memory = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
img_dir = "images"
label_dir = "labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes # only transform the image
        return img, bboxes

transform = Compose([transforms.Resize((488,488)), transforms.ToTensor()])

# training function
# train_loader is those training data that already separated with the bathc_size (by name convention)
def train_function (train_loader, model, optimizer, loss_function):
    loop = tqdm(train_loader, leave=True)
    loss_list = []

    for batch_idx, (image, label) in enumerate(loop):
        images, labels = image.to(device), label.to(device)
        loss = loss_function(images, labels) # to calculate the loss, we put images and those labels inside
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progrss bar
        loss.set_postfix(loss = loss.item()) # tqdm display the loss at the end

    print(f'Mean loss was {sum(loss_list)/len(loss_list)}')

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device) # for the neuralNet _create_fcs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # weight_decay is for drop_out
    loss_function = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(csv_file = "8examples.csv",
                                transform = transform,
                                img_path = img_dir,
                                label_path = label_dir)

    test_dataset = VOCDataset(csv_file= "test.csv",
                                transform = transform,
                                img_path = img_dir,
                                label_path = label_dir)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              pin_memory=pin_memory,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=False) #drop_last is if the output is binary, show in one column

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              pin_memory=pin_memory,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=False)

    for epoch in range (eport_num):
        pred_boxes, target_boxes = get_bboxes(
            train_loader,model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        print(f'Train mAP: {mean_avg_prec}')
        train_function(train_loader, model, optimizer,loss_function)

if __name__ == "__main__":
    main()

