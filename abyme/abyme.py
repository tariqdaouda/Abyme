# class EventHandler(object):
#     """docstring for EventHandler"""
#     def __init__(self):
#         super(EventHandler, self).__init__()
#         self.events = []
    
#     def add(self, event):
#         self.events.append(event)

#     def launch(self, caller):
#         for evt in self.events:
#             if isinstance(evt, _Stage) :
#                 evt.dig(caller)
#             else:
#                 evt(caller)

#     def __call__(self, *args, **kwargs):
#         return self.launch(*args, **kwargs)

#     def __getitem__(self, i):
#         return self.events[i]

#     def __setitem__(self, i, event):
#         self.events[i] = events

# class _Stage(object):
#     """docstring for _Stage"""
#     def __init__(self, event_names=None, *args, **kwargs):
#         super(_Stage, self).__init__()
#         import inspect

#         self.store = {}
#         self.events = {} 
#         if event_names:
#             if type(event_names) is str :
#                 e_names = [event_names]
#             else :
#                 e_names = event_names
#             self.events.update( { event_name: EventHandler() for event_name in e_names} )
        
#         signature = inspect.signature(self._init)
#         key_word_args = {}
#         for i, item in enumerate(signature.parameters.items()):
#             if item[0] not in ["args, "'kwargs'] :
#                 try:
#                     key_word_args[item[0]] = args[i]
#                 except IndexError:
#                     pass
#                 if item[1].default is not inspect.Parameter.empty:
#                     key_word_args[item[0]] = item[1].default
        
#         key_word_args.update(kwargs)
#         if len(key_word_args) > 0 :
#             self._init(**key_word_args)
        
#     def setup(self, *args, **kwargs):
#         self._init(*args, **kwargs)
#         return self

#     def _init(self, *args , **kwargs):
#         pass

#     def dig(self, caller):
#         raise NotImplemented("Must be implemented in child")

#     def more(self, event_name, *stages):
#         for stage in stages :
#             self.events[event_name].add(stage)
#         return self

#     def __call__(self, event_name, *stages):
#         return self.more(event_name, *stages)

#     def __getitem__(self, k):
#         return self.store[k]

#     def __setitem__(self, k, v):
#         self.store[k] = v

# class Ground(_Stage):
#     """docstring for Ground"""
#     def __init__(self, *args, **kwargs):
#         super(Ground, self).__init__("dig")
    
#     def dig(self, caller=None):
#         self.events["dig"](self)

# class IterrationLooper(_Stage):
#     """docstring for IterrationLooper"""
#     def __init__(self, *args, **kwargs):
#         super(IterrationLooper, self).__init__(["start", "iteration_start", "iteration_end", "end"], *args, **kwargs)
    
#     def _init(self, nb_iterations):
#         self["nb_iterations"] = nb_iterations
#         self["current_iteration"] = 0

#     def dig(self, caller):
#         self.events["start"](self)
#         for it in range(self["nb_iterations"]):
#             self.events["iteration_start"](self)
#             self["current_iteration"] += 1
#             self.events["iteration_end"](self)
#         self.events["end"](self)

# class PrintMessage(_Stage):
#     """docstring for PrintMessage"""

#     def __init__(self, *args, **kwargs):
#         super(PrintMessage, self).__init__(None, *args, **kwargs)

#     def _init(self, message):
#         self["message"] = message

#     def dig(self, caller):
#         print(self["message"])

# class PretyPrintStore(_Stage):
#     """docstring for PretyPrintStore"""
#     def _init(self, fields=None):
#         if type(fields) is str :
#             l_fields = [fields]
#         else :
#             l_fields = fields

#         self.store["fields"] = l_fields
        
#     def dig(self, caller):
#         import json
        
#         if self["fields"] is None :
#             store = caller.store
#         else :
#             store = { key: caller[key] for key in self["fields"] }
#         try :
#             print(
#                 json.dumps(
#                     store,
#                     indent=4, sort_keys=True
#                 )
#             )
#         except TypeError:
#             print(caller.store)

# class DataLooper(_Stage):
#     """docstring for DataLooper"""
#     def __init__(self, *args, **kwargs):
#         super(DataLooper, self).__init__(["start"], *args, **kwargs)
    
#     def _init(self, data_loader) :
#        self["data_loader"] = data_loader

#     def dig(self, caller):
#         for data in self["data_loader"]:
#             self["data"] = data
#             self.events["start"](self)

# class SupervisedPass(_Stage):
#     """docstring for SupervisedPass"""
#     def __init__(self, model, optimizer, criterion, update_parameters, inputs_targets_formater):
#         super(SupervisedPass, self).__init__(["start", "end"])
#         self["model"] = model
#         self["optimizer"] = optimizer
#         self["criterion"] = criterion
#         self["update_parameters"] = update_parameters
#         self._inputs_targets_formater = inputs_targets_formater
    
#     @property
#     def inputs_targets_formater(self):
#         return self._inputs_targets_formater
    
#     def dig(self, caller):
#         self.events["start"](self)
#         model_kwargs, targets = self.inputs_targets_formater(caller["data"])
#         self["optimizer"].zero_grad()
#         outputs = self["model"](**model_kwargs)
#         loss = self["criterion"](outputs, targets)
#         self["last_loss"] = loss.item()

#         if self["update_parameters"] :
#             loss.backward()
#             self["optimizer"].step()
        
#         self.events["end"](self)
#         # return self["last_loss"]
    
# if __name__ == '__main__':
    
#     def get_data_loader(train, mask_target, batch_size=64):
#         from torchvision import datasets, transforms
        
#         if mask_target :
#             target_transform = lambda x : 0
#         else :
#             target_transform = None

#         reshape_img = lambda x : torch.reshape(x, (-1, ))

#         mnist_trainset = datasets.MNIST( root='./data', train=train, download=True, transform=transforms.Compose([transforms.ToTensor(), reshape_img]), target_transform=target_transform )
#         loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=not train, num_workers=4)

#         return loader

#     def data_formater(batch_data):
#         return ( {"horus_input": batch_data[0]}, batch_data[0] )

#     import villanitools.utils.models.hgn as HGN
#     model = HGN.HolographicUniverse(
#         horus_input_size = 28*28,
#         horus_layer_size = 128,
#         horus_nb_layers = 2,
#         horus_output_size = 1,
#         horus_non_linearity = "sin",
#         prima_materia_name = "normal",
#         prima_materia_sample_size= 128,
#         prima_materia_nb_functions = 10,
#         horus_midgard_scale = 1,
#         jormungandr_layer_size = 128,
#         jormungandr_nb_layers = 6,
#         jormungandr_output_size = 28*28,
#         jormungandr_non_linearity = "sin",
#         midgard_horus_non_linearity = "sin",
#         prima_materia_kwargs=None
#     )

#     criterion = torch.nn.modules.loss.MSELoss()
#     optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

#     ground = Ground()
#     looper = IterrationLooper()
#     pp_store = PretyPrintStore()
#     data_loader = get_data_loader(True, False)
#     data_looper = DataLooper(data_loader)
#     trainer = SupervisedPass(model, optimizer, criterion, True, data_formater)

#     ground("dig",
#         looper.setup(10)("iteration_start",
#             data_looper(
#                 "start",
#                 trainer("end",
#                     pp_store.setup("last_loss")
#                 )
#             )
#         ).more("end",
#             PrintMessage("the real end")
#         ).more("start",
#             PrintMessage("Training starts")
#         )
#     ).dig()
