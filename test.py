import os
import pandas as pd
import torch
# class getData ():
#     def __init__(self, csv_file, label_path):
#         self.csv_file = pd.read_csv(csv_file)
#         self.label_path = label_path
#
#     def __len__(self):
#         return len(self.csv_file)
#
#
#     def __getitem__(self, item):
#         label_list = []
#         label_path = os.path.join(self.label_path, self.csv_file.iloc[item,1])
#         print(label_path)
#         with open(label_path) as o:
#             for i in o.readlines():
#                 class_label, x,y,w,h = [
#                     float(x) if float(x) != int(float(x)) else float(x)
#                     for x in i.replace('\n','').split()
#                 ]
#                 label_list.append([class_label,x,y,w,h])
#
#         return label_list
#
# label_path = 'labels'
#
# label_list = getData(csv_file='train.csv', label_path=label_path)
# print(type(label_list))
#
# # dataFrame = pd.DataFrame(label_path)
# # print(dataFrame.head(10))

# list = [[[1],[3],[7]],[[2],[4],[5]]]
# print(list[1])
# sored_lost = sorted(list,key=lambda x:x[1], reverse=True)
# print(sored_lost)

# t1 = torch.tensor([[1,2,3]])
# t2 = torch.tensor([4,5,6])
# print(t1.shape)
# t1 = t1.unsqueeze(-1)
# print(t1.shape)

# print(t1.unsqueeze(-1)) # add a dimension
# # t3 = torch.cat((t1,t2),dim=0)
# t4 = torch.cat((t1.unsqueeze(0),t2.unsqueeze(0)),dim=0)
# # print(t3)
# print(t4)
# t5 = torch.tensor([[1],[5]])
# print(t5.argmax(0))
# print(t5.argmax(1))
# print(t5[1])
# print(t5[1].shape)
# t6 = torch.tensor([1,2])
# print(t6.argmax(0))
# a = torch.randn(1,1)
# a = torch.randn(1,2)
# print(a.shape)

# a = torch.tensor([[1,2,3],[3,2,1]])
# print(a)
# print(a.argmax(0))
# print(a.argmax(1))

# a = torch.tensor([[[1],[2],[3]],[[3],[2],[1]]])
# print(a.argmax(1))

# a = torch.tensor([15])
# b = torch.tensor([16])
# c = torch.cat((a,b), dim=0)
# c = c.unsqueeze(-1)
# print(c)
# print(c.shape)

# a = torch.arange(7) # 7 columns
# # print(a)
# b = a.repeat(1,7,1) # repeat 64 times with 7 row
# print(b.shape)
# c = b.unsqueeze(-1)
# print(c)
'''max_range = 7
scale_list = []
x_list = []
unscale = []
for i in range (max_range):
    scale = (i+1)/max_range
    scale_list.append(scale)
    x = scale - (1/max_range)*0.5
    x_list.append(x)
# print(scale_list)
# print(x_list)

for i in range(max_range):
    unscalue_x = 7 * x_list[i]
    unscale.append(unscalue_x)

# print(unscale)
indecing = torch.arange(7)
# print(indecing)

unknow_list = []

# for i in range(max_range):
    # unknow = x_list[...,i] + indecing * 1/7
    # unknow_list.append(unknow)

# print(unknow_list)

# test_tensor = torch.tensor([[(1),(4),(5)],
#               [(9),(6),(3)]])
#
# print(test_tensor[...,[0]])

best_index = torch.arange(3).repeat(3,1).unsqueeze(-1)
print(best_index.shape)
# print(best_index)

x = torch.tensor([[0.5],
                 [0.5],
                 [0.5],])
image = []
for i in range (3):
    image_x = 1/3 * (x[...,:1] + best_index)
    image.append(image_x)

print(image)'''

# list = torch.arange(7).repeat(1,7,1).unsqueeze(-1)
# print(list)
# lsPermute = list.permute(0,2,1,3)
# print(lsPermute)