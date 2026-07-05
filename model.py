import torch
import torch.nn as nn

#TODO: Test that the model outputs a tensor of the correct shape

class YOLOModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        # All layers but the last use a Leakly ReLu w/ Neg Slope = 0.1 
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(192, 128, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        # Block 4
        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 256, 1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 256, 1)
        self.conv12 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 256, 1)
        self.conv14 = nn.Conv2d(256, 512, 3, padding=1)

        self.conv15 = nn.Conv2d(512, 512, 1)
        self.conv16 = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # Block 5
        self.conv17 = nn.Conv2d(1024, 512, 1)
        self.conv18 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv19 = nn.Conv2d(1024, 512, 1)
        self.conv20 = nn.Conv2d(512, 1024, 3, padding=1)

        self.conv21 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv22 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)

        # Block 6
        self.conv23 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv24 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Fully Connected
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 30 * 7 * 7)

    def forward(self, x):

        # Block 1
        x = self.act(self.conv1(x))
        x = self.pool1(x)

        # Block 2
        x = self.act(self.conv2(x))
        x = self.pool2(x)

        # Block 3
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.pool3(x)

        # Block 4
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x))
        x = self.act(self.conv9(x))
        x = self.act(self.conv10(x))
        x = self.act(self.conv11(x))
        x = self.act(self.conv12(x))
        x = self.act(self.conv13(x))
        x = self.act(self.conv14(x))
        x = self.act(self.conv15(x))
        x = self.act(self.conv16(x))
        x = self.pool4(x)

        # Block 5
        x = self.act(self.conv17(x))
        x = self.act(self.conv18(x))
        x = self.act(self.conv19(x))
        x = self.act(self.conv20(x))
        x = self.act(self.conv21(x))
        x = self.act(self.conv22(x))

        # Block 6
        x = self.act(self.conv23(x))
        x = self.act(self.conv24(x))

        x = torch.flatten(x, start_dim=1) # because of batching

        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x
    
#TODO: Test + Debug