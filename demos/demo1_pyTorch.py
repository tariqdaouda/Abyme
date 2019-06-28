import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    
    import abyme

    def get_data_loader(train, mask_target, batch_size=64):
        from torchvision import datasets, transforms
        
        if mask_target :
            target_transform = lambda x : 0
        else :
            target_transform = None

        reshape_img = lambda x : torch.reshape(x, (-1, ))

        mnist_trainset = datasets.MNIST( root='./data', train=train, download=True, transform=transforms.Compose([transforms.ToTensor(), reshape_img]), target_transform=target_transform )
        loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=not train, num_workers=4)

        return loader

    def data_formater(batch_data):
        return ( {"horus_input": batch_data[0]}, batch_data[1] )

    model = TheModelClass()

    criterion = torch.nn.modules.loss.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    ground = abyme.base.Ground()
    looper = abyme.base.IterrationLooper()
    pp_store = abyme.base.PretyPrintStore()
    data_loader = get_data_loader(True, False)
    data_looper = abyme.base.DataLooper(data_loader)
    trainer = abyme.pyTorch.SupervisedPass(model, optimizer, criterion, True, data_formater)

    ground("dig",
        looper.setup(10)("iteration_start",
            data_looper(
                "start",
                trainer("end",
                    pp_store.setup("last_loss")
                )
            )
        ).more("end",
            PrintMessage("the real end")
        ).more("start",
            PrintMessage("Training starts")
        )
    ).dig()
