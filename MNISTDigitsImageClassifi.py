#matplotlib inline
import matplotlib
import helper
import torch
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

# Define a transform to normalize the data
class MNIST_DigitsImageClassifier(nn.Module):
    def __init__(self, imgpath='~/.pytorch/MNIST_data/'):
        super().__init__()
        #input layer
        self.model = nn.Sequential(OrderedDict([('fc1',nn.Linear(784,128)), ('relu1', nn.ReLU()), ('fc2',nn.Linear(128,64)), ('relu2', nn.ReLU()), ('fc3', nn.Linear(64,10)), ('log_softmax', nn.LogSoftmax(dim=1))]))

        self.criterion = nn.NLLLoss()
        self.imagepath = imgpath
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.003)

    def preprocessImage(self):
        #Loading and preprocessing image
        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, )),
                                  ])
        # Download and load the training data
        self.trainset = datasets.MNIST(self.imagepath, download=True, train=True, transform=self.transform)
        #trainloader has the normalized image with batch size 64)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)



    def trainnetwork(self,ep=5):
        epochs = ep
        self.preprocessImage()
        for e in range(epochs):
            running_loss = 0

            for images, labels in self.trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                output = self.model(images)

                self.optimizer.zero_grad()
                loss = self.criterion(output,labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(self.trainloader)}")



    def testnetwork(self):
        images, labels = next(iter(self.trainloader))

        img = images[0].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = self.model(img)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        helper.view_classify(img.view(1, 28, 28), ps)

#    def main(imgpath, ep):
#        print("Predicting MNIST images using Pytorch neural networks")
#        network = MNIST_Pytorch(imgpath)
#        network.trainnetwork(ep)
#        network.testnetwork()


if __name__ == '__main__':
    imgpath = '~/.pytorch/MNIST_data/'
    ep = 5
#    main(imgpath, ep)
    print("Predicting MNIST images using Pytorch neural networks")
    network = MNIST_DigitsImageClassifier(imgpath)
    network.trainnetwork(ep)
    network.testnetwork()
