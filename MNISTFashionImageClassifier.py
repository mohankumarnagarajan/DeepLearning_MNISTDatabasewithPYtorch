import matplotlib
import helper
import torch
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

# Define a transform to normalize the data
class MNIST_FashionImageClassifier(nn.Module):
    def __init__(self, imgpath='~/.pytorch/MNIST_data/'):
        super().__init__()
        #input layer
        self.n_input = 784
        self.n_hidden = [256,128,64]
        self.n_output = 10
        self.model = nn.Sequential((OrderedDict([('fc1', nn.Linear(self.n_input,self.n_hidden[0])), ('relu1',nn.ReLU()), ('fc2',nn.Linear(self.n_hidden[0],self.n_hidden[1])), ('relu2',nn.ReLU()), ('fc3',nn.Linear(self.n_hidden[1],self.n_hidden[2])), ('relu3',nn.ReLU()), ('fc4',nn.Linear(self.n_hidden[2],self.n_output)), ('Log_softmax',nn.LogSoftmax(dim=1))])))

        self.criterion = nn.NLLLoss()
        self.imagepath = imgpath
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.003)

    def preprocessImage(self):
        #Loading and preprocessing image
        # Define a transform to normalize the data
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        # Download and load the training data
        self.trainset = datasets.FashionMNIST(self.imagepath, download=True, train=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)

        # Download and load the test data
        self.testset = datasets.FashionMNIST(self.imagepath, download=True, train=False, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=True)

    def trainnetwork(self,ep=10):
        epochs = ep
        self.preprocessImage()
        epochs = 10
        for e in range(epochs):
            cumulative_loss = 0
            for images,labels in self.trainloader:
                images = images.reshape(images.shape[0],-1)
                output = self.model(images)
                self.optimizer.zero_grad()
                loss = self.criterion(output,labels)
                loss.backward()
                self.optimizer.step()

                cumulative_loss += loss.item()
            else:
                print(f"Training Loss for epoch {cumulative_loss/len(self.trainloader)}")


    def testnetwork(self):
        images, labels = next(iter(self.trainloader))

        img = images[0].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = self.model(img)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        helper.view_classify(img.view(1, 28, 28), ps)



if __name__ == '__main__':
    imgpath = '~/.pytorch/F_MNIST_data/'
    ep = 5
#    main(imgpath, ep)
    print("Predicting MNIST Fashion images using Pytorch neural networks")
    network = MNIST_FashionImageClassifier(imgpath)
    network.trainnetwork(ep)
    network.testnetwork()
