from . import abstract
import torch
import os

class SaveModel(abstract._Stage):
    """Save a pyTorch model"""
    def __init__(self, *args, **kwargs):
        super(SaveModel, self).__init__(["before_save", "after_save"], *args, **kwargs)

    def _init(self, model, base_filename, optimizer=None, loss=None, folder=".", extension=".pyTorch", overwrite=False, prefix_callable=None):
        if extension[0] != "." :
            ext = "."+extension
        else :
            ext = extension         
        
        if base_filename.find(ext) < 0 :
            self["base_filename"] = base_filename + ext
        else :
            self["base_filename"] = base_filename

        self["prefix"] = prefix_callable
        self["folder"] = folder
        self["model"] = model
        self["optimizer"] = optimizer
        self["loss"] = loss
        self['overwrite'] = overwrite
    
    def dig(self, caller):
        self.events["before_save"](self)

        if not self["overwrite"]:
            import time
            fix = time.ctime().replace(" ", "-") + "_"
            base_filename = fix + self["base_filename"]

        if self["prefix"]:
            base_filename = "%s%s" % (abstract.call_if_callable(self["prefix"]), base_filename)

        if not os.path.isdir(self["folder"]):
             os.mkdir(self["folder"])

        filename = os.path.join(self["folder"], base_filename)

        res = {
            "model_state_dict": self["model"].state_dict()      
        }
        if self["optimizer"] :
            res["optimizer"] = self["optimizer"].state_dict()

        if self["loss"]:
          res["loss"] = self["loss"]

        torch.save(
            res,
            filename
        )
        
        self.events["after_save"](self)

class SupervisedPass(abstract._Stage):
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
