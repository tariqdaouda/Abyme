import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    
    import abyme.base as AB
    import abyme.pyTorch as AP

    def get_data_loader(train, mask_target, batch_size=64):
        from torchvision import datasets, transforms
        
        if mask_target :
            target_transform = lambda x : 0
        else :
            target_transform = None

        mnist_trainset = datasets.MNIST( root='./data', train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=target_transform )
        loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=not train, num_workers=4)

        return loader

    def data_formater(batch_data):
        return ( {"x": batch_data[0]}, batch_data[1] )

    model = TheModelClass()

    criterion = torch.nn.modules.loss.F.nll_loss
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    ground = AB.Ground()
    looper = AB.IterrationLooper()
    pp_store = AB.PrettyPrintStore()
    data_loader = get_data_loader(True, False)
    data_looper = AB.DataLooper(data_loader)
    trainer = AP.SupervisedPass(model, optimizer, criterion, True, data_formater)

    ground("dig",
        looper.setup(10)("iteration_start",
            data_looper(
                "iteration_end",
                trainer("end",
                    AB.PeriodicTrigger(100)("trigger",
                        pp_store.setup("last_loss")
                    ),
                    AB.PeriodicTrigger(1)("trigger",
                        AB.CSVStreamSaver(filemame="test.csv", prefix="DEMO_", select_fields=["last_loss"], pandas_to_csv_kwargs = {"sep": ";"})
                    )
                )
            )
        ).more("end",
            AB.PrintMessage("the real end")
        ).more("start",
            AB.PrintMessage("Training starts")
        )
    ).dig()
