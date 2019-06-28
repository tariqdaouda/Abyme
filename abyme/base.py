from . abstract import _Stage

def filter_store(store, fields=None):
    if fields is None :
        return store
    else :
        return{ key: store[key] for key in fields }

class Ground(_Stage):
    """docstring for Ground"""
    def __init__(self, *args, **kwargs):
        super(Ground, self).__init__("dig")
    
    def dig(self, caller=None):
        self.events["dig"](self)
        # print("aaasss", self.events)

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

class PeriodicTrigger(_Stage):
    """docstring for PeriodicTrigger"""

    def __init__(self, *args, **kwargs):
        super(PeriodicTrigger, self).__init__(["trigger"], *args, **kwargs)

    def _init(self, period):
        self["counter"] = 0
        self["period"] = period

    def dig(self, caller):
        if self['counter'] % self["period"] == 0 : 
            self.events["trigger"](caller)
        self['counter'] += 1

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
        
        store = filter_store(caller.store, self["fields"])
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

class CSVStreamSaver(_Stage):
    """docstring for CSVStreamSaver"""
    def __init__(self, *args, **kwargs):
        super(CSVStreamSaver, self).__init__(None, *args, **kwargs)
    
    def _init(self, filemame, prefix=None, select_fields=None, header=True, pandas_to_csv_kwargs=None):
        self["filemame"] = filemame
        if prefix is None:
            self["prefix"] = ""
        else :
            self["prefix"] = prefix
        self["select_fields"] = select_fields
        self["line_number"] = 0
        self["header"] = header

        if pandas_to_csv_kwargs is None :
            self.pandas_to_csv_kwargs = {}
        else :
            self.pandas_to_csv_kwargs = pandas_to_csv_kwargs    
            for name in ["header", "index"]:
                if name in self.pandas_to_csv_kwargs:
                    raise ValueError("Pandas argument %s has predefided and cannot be changed" % name)
        
        self.file = open(filemame, "w")

    def dig(self, caller):
        import pandas as pd
        store = filter_store(caller, self["select_fields"])
        store = { "%s%s" % (self["prefix"], key) : [value] for key, value in store.items() }
        
        store["id"] = [self["line_number"]]

        df = pd.DataFrame.from_dict(store)
        df.to_csv(self.file, header=self["header"], index=False, **self.pandas_to_csv_kwargs)
        self["line_number"] += 1
