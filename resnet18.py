from libraries import *
from torch.optim.lr_scheduler import StepLR

# ResNet18 model
model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
ResNet18 = model.to(device) 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  #(set loss function)
optimizer = optimizer.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
# scheduler for dynamically adjusting the learning rate
scheduler = StepLR(optimizer, step_size=2,gamma=0.1)
