# WORKING! 

import torch
import torch.nn as nn

# (kernal_size, num_filters, stride, padding)
ARCHITECTURE_CONFIG = [
    (7, 64, 2, 3),
    "M", # maxpool
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 0),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # repeat 4 times!
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
    # Note: this does not include FC layer
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # bias uselss with batchnorm
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = ARCHITECTURE_CONFIG
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple: # single use conv layer
                layers += [
                    CNNBlock(
                        in_channels, 
                        x[1], 
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str: # maxpool
                layers += [
                    nn.MaxPool2d(
                        kernel_size=2, stride=2
                    )
                ]
            
            elif type(x) == list: # repeated layers
                repeated_layers = x[:-1]
                num_repeat = x[-1]

                for _ in range(num_repeat):
                    for layer in repeated_layers:
                        kernel_size, out_channels, stride, padding = layer

                        layers += [
                            CNNBlock(
                                in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        ]

                        # Ensure next layer's input matches this layer's output 
                        in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    # Default for VOC
    def _create_fcs(self, split_size=7, num_boxes=2, num_classes=20):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # should be 4096 => change based on compute power
            nn.Dropout(0.0), # if model is overfitting, change this!
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)) # (S, S, 30) for VOC
        )      

if __name__ == "__main__":
    print("Testing Model.py...")
    model = Yolov1()
    x = torch.randn((2, 3, 448, 448)) # raw image
    print(model(x).shape)

""" 
This is my first, unassisted attempt at creating the model architecture
- Very redundant
- Hard-coded dimensions makes it harder to extend to other datasets
- Very susceptible to silent errors, not good programming practice
- No batch normalization
"""
# class YOLOModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#         # All layers but the last use a Leakly ReLu w/ Neg Slope = 0.1 
#         self.act = nn.LeakyReLU(negative_slope=0.1)

#         # Block 1
#         self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
#         self.pool1 = nn.MaxPool2d(2, stride=2)

#         # Block 2
#         self.conv2 = nn.Conv2d(64, 192, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, stride=2)

#         # Block 3
#         self.conv3 = nn.Conv2d(192, 128, 1)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv5 = nn.Conv2d(256, 256, 1)
#         self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, stride=2)
        
#         # Block 4
#         self.conv7 = nn.Conv2d(512, 256, 1)
#         self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
#         self.conv9 = nn.Conv2d(512, 256, 1)
#         self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
#         self.conv11 = nn.Conv2d(512, 256, 1)
#         self.conv12 = nn.Conv2d(256, 512, 3, padding=1)
#         self.conv13 = nn.Conv2d(512, 256, 1)
#         self.conv14 = nn.Conv2d(256, 512, 3, padding=1)

#         self.conv15 = nn.Conv2d(512, 512, 1)
#         self.conv16 = nn.Conv2d(512, 1024, 3, padding=1)
#         self.pool4 = nn.MaxPool2d(2, stride=2)

#         # Block 5
#         self.conv17 = nn.Conv2d(1024, 512, 1)
#         self.conv18 = nn.Conv2d(512, 1024, 3, padding=1)
#         self.conv19 = nn.Conv2d(1024, 512, 1)
#         self.conv20 = nn.Conv2d(512, 1024, 3, padding=1)

#         self.conv21 = nn.Conv2d(1024, 1024, 3, padding=1)
#         self.conv22 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)

#         # Block 6
#         self.conv23 = nn.Conv2d(1024, 1024, 3, padding=1)
#         self.conv24 = nn.Conv2d(1024, 1024, 3, padding=1)

#         # Fully Connected
#         self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
#         self.fc2 = nn.Linear(4096, 30 * 7 * 7)

#     def forward(self, x):

#         # Block 1
#         x = self.act(self.conv1(x))
#         x = self.pool1(x)

#         # Block 2
#         x = self.act(self.conv2(x))
#         x = self.pool2(x)

#         # Block 3
#         x = self.act(self.conv3(x))
#         x = self.act(self.conv4(x))
#         x = self.act(self.conv5(x))
#         x = self.act(self.conv6(x))
#         x = self.pool3(x)

#         # Block 4
#         x = self.act(self.conv7(x))
#         x = self.act(self.conv8(x))
#         x = self.act(self.conv9(x))
#         x = self.act(self.conv10(x))
#         x = self.act(self.conv11(x))
#         x = self.act(self.conv12(x))
#         x = self.act(self.conv13(x))
#         x = self.act(self.conv14(x))
#         x = self.act(self.conv15(x))
#         x = self.act(self.conv16(x))
#         x = self.pool4(x)

#         # Block 5
#         x = self.act(self.conv17(x))
#         x = self.act(self.conv18(x))
#         x = self.act(self.conv19(x))
#         x = self.act(self.conv20(x))
#         x = self.act(self.conv21(x))
#         x = self.act(self.conv22(x))

#         # Block 6
#         x = self.act(self.conv23(x))
#         x = self.act(self.conv24(x))

#         x = torch.flatten(x, start_dim=1) # because of batching

#         x = self.act(self.fc1(x))
#         x = self.fc2(x)

#         return x
