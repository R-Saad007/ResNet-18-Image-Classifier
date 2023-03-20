from libraries import *
from torch.optim.lr_scheduler import StepLR

# ResNet18 (pretrained) model
ResNet18 = torchvision.models.resnet18(pretrained=True)
ResNet18 = ResNet18.to(device) 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  #(set loss function)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# scheduler for dynamically adjusting the learning rate
scheduler = StepLR(optimizer, step_size=2,gamma=0.1)
