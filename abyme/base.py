from . abstract import _Stage

class Ground(_Stage):
    """docstring for Ground"""
    def __init__(self, *args, **kwargs):
        super(Ground, self).__init__("dig")
    
    def dig(self, caller=None):
        self.events["dig"](self)

class IterrationLooper(_Stage):
    """docstring for IterrationLooper"""
    def __init__(self, *args, **kwargs):
        super(IterrationLooper, self).__init__(["start", "iteration_start", "iteration_end", "end"], *args, **kwargs)
    
    def _init(self, nb_iterations):
        self["nb_iterations"] = nb_iterations
        self["current_iteration"] = 0

    def dig(self, caller):
        self.events["start"](self)
        for it in range(self["nb_iterations"]):
            self.events["iteration_start"](self)
            self["current_iteration"] += 1
            self.events["iteration_end"](self)
        self.events["end"](self)

class PrintMessage(_Stage):
    """docstring for PrintMessage"""

    def __init__(self, *args, **kwargs):
        super(PrintMessage, self).__init__(None, *args, **kwargs)

    def _init(self, message):
        self["message"] = message

    def dig(self, caller):
        print(self["message"])

class PrettyPrintStore(_Stage):
    """docstring for PretyPrintStore"""
    def _init(self, fields=None):
        if type(fields) is str :
            l_fields = [fields]
        else :
            l_fields = fields

        self.store["fields"] = l_fields
        
    def dig(self, caller):
        import json
        
        if self["fields"] is None :
            store = caller.store
        else :
            store = { key: caller[key] for key in self["fields"] }
        try :
            print(
                json.dumps(
                    store,
                    indent=4, sort_keys=True
                )
            )
        except TypeError:
            print(caller.store)

class DataLooper(_Stage):
    """docstring for DataLooper"""
    def __init__(self, *args, **kwargs):
        super(DataLooper, self).__init__(["start", "iteration_start", "iteration_end", "end"], *args, **kwargs)
    
    def _init(self, data_loader) :
       self["data_loader"] = data_loader

    def dig(self, caller):
        self.events["start"](self)
        for data in self["data_loader"]:
            self.events["iteration_start"](self)
            self["data"] = data
            self.events["iteration_end"](self)
        self.events["end"](self)
