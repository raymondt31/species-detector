# Working!

import torchvision
import torch
import torchvision.transforms as transforms

class VOCdataset(torch.utils.data.Dataset):
    # TODO: edit to take in S,B,C parameters
    def __init__(self, S=7, B=2, C=20, transform=None):
        self.S = S
        self.B = B
        self.C = C

        # Hardcoded resize replaced with transform to facilitate data augmentation later
        self.transform = transform

        self.raw_dataset = torchvision.datasets.VOCDetection(
            root="./data", year="2012", image_set="train", download=False # set this to true when first cloned to Colab 
        )
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
    
    # this gets triggered by [] notation; ie dataset[0] = dataset.__getitem__(0)
    def __getitem__(self, idx):

        raw_img, annotation = self.raw_dataset[idx]

        boxes = []        
        for object in annotation['annotation']['object']: 
            
            raw_coords = object['bndbox']
            name = object['name']
            class_label = self.obj_to_index[name]

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

            # Append as [class, x, y, w, h]
            boxes.append([class_label, n_xcenter, n_ycenter, n_width, n_height])

        # apply transformation if passed in; note that we now transform before defining the target tensor
        if self.transform:
            img_tensor, boxes = self.transform(raw_img, boxes)
        else: # otherwise just turn the raw image into a usable tensor
            fallback_transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor()
            ])
            img_tensor = fallback_transform(raw_img)

        target_tensor = torch.zeros((self.S, self.S, self.C + self.B * 5))
        
        for box in boxes: 
            class_label, n_xcenter, n_ycenter, n_width, n_height = box
            cell_col = int(n_xcenter * self.S)
            cell_row = int(n_ycenter * self.S)

            x_center = (n_xcenter * self.S) - cell_col
            y_center = (n_ycenter * self.S) - cell_row

            # Make sure cell doesn't already have an object detected
            if target_tensor[cell_row, cell_col, self.C] == 0:

                # each tensor input is [c1, c2, ... , c20, p_c1, x, y, w, h, p_c2, x, y, w, h]
                #                                           20  21 22 23 24
                target_tensor[cell_row, cell_col, class_label] = 1
                target_tensor[cell_row, cell_col, self.C] = 1

                box_coords = torch.tensor([x_center, y_center, n_width, n_height])
                target_tensor[cell_row, cell_col, self.C+1: self.C+5] = box_coords

        return img_tensor, target_tensor
    
#TODO: Test + Debug
if __name__ == "__main__":
    print("Testing Dataset.py...")

    dataset = VOCdataset(transform=None)

    img, target = dataset[31]

    print(f"\nImg Tensor Shape: {img.shape}, Expected: [3, 448, 448]")
    print(f"Target Tensor Shape: {target.shape}, Expected: [7, 7, 30]")

    # limit search to cells with objects in them
    object_cells = (target[..., 20] == 1).nonzero(as_tuple=False)
    print(f"Found {len(object_cells)} object(s)")

    for cell in object_cells:
        row, col = cell[0].item(), cell[1].item()

        cell_data = target[row, col]
        class_idx = torch.argmax(cell_data[:20]).item()
        confidence = cell_data[20].item()
        x, y, w, h = cell_data[21:25].tolist()

        assert 0 <= x <= 1, "x offset out of bounds!"
        assert 0 <= y <= 1, "y offset out of bounds!"
        assert 0 <= w <= 1, "width out of bounds!"
        assert 0 <= h <= 1, "height out of bounds!"

        print(f"\n--- Object in Cell (Row {row}, Col {col}) ---")
        print(f"Class Index: {class_idx}")
        print(f"Confidence:  {confidence}")
        print(f"Box (x, y):  ({x:.4f}, {y:.4f}")
        print(f"Box (w, h):  ({w:.4f}, {h:.4f})")
    
    print("\nDone Testing!")