from . abstract import _Stage

class SaveModel(_Stage):

class SupervisedPass(_Stage):
    """docstring for SupervisedPass"""
    def __init__(self, *args, **kwargs):
        super(SupervisedPass, self).__init__(["start", "end"], *args, **kwargs)
    
    def _init(self, model, optimizer, criterion, update_parameters, inputs_targets_formater):
        self["model"] = model
        self["optimizer"] = optimizer
        self["criterion"] = criterion
        self["update_parameters"] = update_parameters
        self._inputs_targets_formater = inputs_targets_formater
    
    @property
    def inputs_targets_formater(self):
        return self._inputs_targets_formater
    
    def dig(self, caller):
        self.events["start"](self)
        model_kwargs, targets = self.inputs_targets_formater(caller["data"])
        self["optimizer"].zero_grad()
        outputs = self["model"](**model_kwargs)
        loss = self["criterion"](outputs, targets)
        self["last_loss"] = loss.item()

        if self["update_parameters"] :
            loss.backward()
            self["optimizer"].step()
        
        self.events["end"](self)
