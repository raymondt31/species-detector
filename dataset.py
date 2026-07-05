import torchvision
import torch
from PIL import Image

DIM = 448
IMAGE_RES = (DIM, DIM)

# resize img to 448x448
def resize(img_obj):
    resize = img_obj.resize(IMAGE_RES)
    img = torchvision.transforms.ToTensor()(resize)
    return img

class VOCdataset(torch.utils.data.Dataset):
    # TODO: edit to take in S,B,C parameters
    def __init__(self, S=7, B=2, C=20):
        self.S = S
        self.B = B
        self.C = C
        self.raw_dataset = torchvision.datasets.VOCDetection(root="./data", year="2012", image_set="train", download=False)
        self.obj_to_index = {
            'horse': 0, 'person': 1, 'bottle': 2,
            'dog': 3, 'tvmonitor': 4, 'car': 5,
            'aeroplane': 6, 'bicycle': 7,
            'boat': 8, 'chair': 9, 'diningtable': 10,
            'pottedplant': 11, 'train': 12, 'cat': 13,
            'sofa': 14, 'bird': 15, 'sheep': 16,
            'motorbike': 17, 'bus': 18, 'cow': 19
        }

    # to keep track of epochs
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, idx):

        target_tensor = torch.zeros((7,7,30))
        raw_img, annotation = self.raw_dataset[idx]
        img_tensor = resize(raw_img)

        for object in annotation['annotation']['object']: 
            
            raw_coords = object['bndbox']
            name = object['name']

            # compute normalized width and height
            raw_height = float(raw_coords['ymax']) - float(raw_coords['ymin'])
            raw_width = float(raw_coords['xmax']) - float(raw_coords['xmin'])
            n_height = raw_height / raw_img.size[1]
            n_width = raw_width / raw_img.size[0]

            # compute raw center coordinates
            raw_xcenter = (float(raw_coords['xmin']) + float(raw_coords['xmax'])) / 2
            raw_ycenter = (float(raw_coords['ymin']) + float(raw_coords['ymax'])) / 2

            # compute normalized center coordinates
            n_xcenter = raw_xcenter / raw_img.size[0]
            n_ycenter = raw_ycenter / raw_img.size[1]

            cell_col = int(n_xcenter * 7)
            cell_row = int(n_ycenter * 7)

            x_center = (n_xcenter * 7) - cell_col
            y_center = (n_ycenter * 7) - cell_row
            
            # each tensor input is [c1, c2, ... , c20, p_c1, x, y, w, h, p_c2, x, y, w, h]
            #                                           20  21 22 23 24
            cell_tensor = target_tensor[cell_row][cell_col]
            cell_tensor[self.obj_to_index[name]] = 1
            cell_tensor[20] = 1
            cell_tensor[21] = x_center
            cell_tensor[22] = y_center
            cell_tensor[23] = n_width
            cell_tensor[24] = n_height

        return img_tensor, target_tensor
    
#TODO: Test + Debug