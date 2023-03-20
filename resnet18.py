from libraries import *

# ResNet18 model
ResNet18 = torchvision.models.resnet18(pretrained=True)
ResNet18 = ResNet18.to(device) 
criterion = nn.CrossEntropyLoss()  #(set loss function)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
