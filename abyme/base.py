from . abstract import _Stage
import numpy

def filter_store(store, fields=None):
    if fields is None :
        return store
    else :
        return{ key: store[key] for key in fields }

def flatten_dict(self, dct, separator=".", res = None):
    if res is None :
        res = {}

    for k, v in dct.items():
        if type(v) is dict :
            filter_store[k] = 
        else :
            res[k] = v

def apply_list_transform(value, transforms):
    if transforms is None :
        return value

    val = value
    for trans in transforms:
        val = trans(val)
    return val

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
    
    def _init(self, filemame, fields_tranforms=None, values_tranforms=None, prefix=None, select_fields=None, header=True, pandas_to_csv_kwargs=None):
        self["filemame"] = filemame
        if prefix is None:
            self["prefix"] = ""
        else :
            self["prefix"] = prefix
        self["select_fields"] = select_fields
        self["line_number"] = 0
        self["header"] = header

        self.fields_tranforms = fields_tranforms
        self.values_tranforms = values_tranforms

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
        tmp_store = filter_store(caller, self["select_fields"])
        
        store = {}
        for key, value in tmp_store.items():
            kkey = apply_list_transform(key, self.fields_tranforms)
            vvalue = apply_list_transform(value, self.values_tranforms)
            store["%s%s" % (self["prefix"], kkey)] = [vvalue]
        
        store["id"] = [self["line_number"]]

        df = pd.DataFrame.from_dict(store)
        df.to_csv(self.file, header=self["header"], index=False, **self.pandas_to_csv_kwargs)
        self["line_number"] += 1

    def close(self):
        self.file.close()

class ThresholdTrigger(_Stage):

class NewExtremumTrigger(_Stage):

class NewLowTrigger(_Stage):

class NewHighTrigger(_Stage):

class MovingStats(_Stage):
    """docstring for MovingStats"""
    def __init__(self, *args, **kwargs):
        super(MovingStats, self).__init__(["end"], *args, **kwargs)
    
    def _init(self, caller_field, window_size=100):
        self["values"] = numpy.zeros(window_size)
        self["caller_field"] = caller_field
        self.counter = 0

    def dig(self, caller):
        self["values"][self.counter % len(self["values"]) ] = caller[self["caller_field"]]
        self["average"] = numpy.mean(self["values"])
        self["std"] = numpy.std(self["values"])
        self.counter += 1
        self.events["end"](self)

class Stats(_Stage):
    """docstring for Stats"""
    def __init__(self, *args, **kwargs):
        super(Stats, self).__init__(["end"], *args, **kwargs)
    
    def _init(self, caller_field):
        self["values"] = []
        self["caller_field"] = caller_field
        
    def dig(self, caller):
        self["values"] .append( caller[self["caller_field"]] )
        self["average"] = numpy.mean(self["values"])
        self["std"] = numpy.std(self["values"])
        self.events["end"](self)

class StoreAggregator(_Stage):
    """docstring for StoreAggregator"""
    def __init__(self, *args, **kwargs):
        super(StoreAggregator, self).__init__(["end"], *args, **kwargs)
    
    def _init(self, select_fields, prefix):
        self["select_fields"] = select_fields
        self["prefix"] = prefix
        self["aggregate"] = {}

    def aggregate(self, caller):
        self["aggregate"].update(
            filter_store(caller, self["select_fields"])
        )

    def dig(self, caller):
        self["values".append( caller[self["caller_field"]] )
        self["average"] = numpy.mean(self["values"])
        self["std"] = numpy.std(self["values"])
        self.events["end"](self)

