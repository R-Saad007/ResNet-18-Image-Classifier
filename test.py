from libraries import *
from normalize_load_data import test_dataloader
from resnet18 import *

# Test function
def test(save_flag):
    correct = 0
    accuracy = 0
    total = 0
    ResNet18.eval()
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = ResNet18(images)
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # accuracy
    accuracy = (correct / total) * 100
    print('Finished Testing')
    print('Accuracy on the 10000 test images: %.2f %%' % accuracy)
    print("-----------------------------------------")
    if save_flag:
        # to save the CNN model
        PATH = './cifar_net.pth'
        torch.save(ResNet18.state_dict(), PATH)
        print('Model Saved')
    return accuracy
