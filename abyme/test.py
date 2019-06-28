if __name__ == '__main__':
    
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
        return ( {"horus_input": batch_data[0]}, batch_data[0] )

    import villanitools.utils.models.hgn as HGN
    model = HGN.HolographicUniverse(
        horus_input_size = 28*28,
        horus_layer_size = 128,
        horus_nb_layers = 2,
        horus_output_size = 1,
        horus_non_linearity = "sin",
        prima_materia_name = "normal",
        prima_materia_sample_size= 128,
        prima_materia_nb_functions = 10,
        horus_midgard_scale = 1,
        jormungandr_layer_size = 128,
        jormungandr_nb_layers = 6,
        jormungandr_output_size = 28*28,
        jormungandr_non_linearity = "sin",
        midgard_horus_non_linearity = "sin",
        prima_materia_kwargs=None
    )

    criterion = torch.nn.modules.loss.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    ground = Ground()
    looper = IterrationLooper()
    pp_store = PretyPrintStore()
    data_loader = get_data_loader(True, False)
    data_looper = DataLooper(data_loader)
    trainer = SupervisedPass(model, optimizer, criterion, True, data_formater)

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
