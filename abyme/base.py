from . import abstract
from . import utils
from . import exceptions

import numpy

class Ground(abstract._Stage):
    """docstring for Ground"""
    def __init__(self, *args, **kwargs):
        super(Ground, self).__init__("dig", "final")
    
    def dig(self, caller=None):
        try:
            self.events["dig"](self)        
        except exceptions.EndOfEverything as e:
            print("Ended because:", e)
            self.events["final"](self)        
            

class IterrationLooper(abstract._Stage):
    """docstring for IterrationLooper"""
    def __init__(self, *args, **kwargs):
        super(IterrationLooper, self).__init__(["start", "iteration_start", "iteration_end", "end"], *args, **kwargs)
    
    def _init(self, nb_iterations):
        self["nb_iterations"] = nb_iterations
        self["counter"] = 0

    def dig(self, caller):
        self.events["start"](self)
        for it in range(self["nb_iterations"]):
            self.events["iteration_start"](self)
            self["counter"] += 1
            self.events["iteration_end"](self)
        self.events["end"](self)

class DataLooper(abstract._Stage):
    """docstring for DataLooper"""
    def __init__(self, *args, **kwargs):
        super(DataLooper, self).__init__(["start", "iteration_start", "iteration_end", "end"], *args, **kwargs)
    
    def _init(self, data_loader) :
        self["data_loader"] = data_loader
        self["counter"] = None
    
    def dig(self, caller):
        self.events["start"](self)
        for i, data in enumerate(self["data_loader"]):
            self["counter"] = i
            self.events["iteration_start"](self)
            self["data"] = data
            self.events["iteration_end"](self)
        self.events["end"](self)

class If(abstract._Stage):
    """docstring for PeriodicTrigger"""
    def __init__(self, *args, **kwargs):
        super(If, self).__init__(["dig"], *args, **kwargs)

    def _init(self, condition):
        self["condition"] = condition

    def dig(self, caller):
        try :
            val = self["condition"]()
        except TypeError :
            val = self["condition"]

        if val : 
            self.events["dig"](caller)


class PeriodicTrigger(abstract._Stage):
    """docstring for PeriodicTrigger"""
    def __init__(self, *args, **kwargs):
        super(PeriodicTrigger, self).__init__(["dig"], *args, **kwargs)

    def _init(self, period, wait_periods=0):
        self["counter"] = 0
        self["wait_periods"] = wait_periods
        self["period"] = period

    def dig(self, caller):
        if self["counter"] >= (self["period"] * self["wait_periods"]) and self['counter'] % self["period"] == 0 : 
            self.events["dig"](caller)
        self['counter'] += 1

class ThresholdTrigger(abstract._Stage):
    """"""
    def __init__(self, *args, **kwargs):
        super(ThresholdTrigger, self).__init__(["dig"], *args, **kwargs)

    def _init(self, caller_field, threshold):
        self["caller_field"] = caller_field
        self["threshold"] = threshold

    def dig(self, caller):
        if caller[self["caller_field"]] > self["threshold"] : 
            self.events["dig"](caller)

class NewLowTrigger(abstract._Stage):
    """"""
    def __init__(self, *args, **kwargs):
        super(NewLowTrigger, self).__init__(["dig"], *args, **kwargs)

    def _init(self, caller_field, epsilon=1e-8):
        self["caller_field"] = caller_field
        self["epsilon"] = epsilon
        self["min"] = None

    def dig(self, caller):
        if not self["min"] or self["min"] > caller[self["caller_field"]]:
            self["min"] = caller[self["caller_field"]]
            self.events["dig"](caller)

class NewHighTrigger(abstract._Stage):
    """"""
    def __init__(self, *args, **kwargs):
        super(NewHighTrigger, self).__init__(["dig"], *args, **kwargs)

    def _init(self, caller_field, epsilon=1e-8):
        self["caller_field"] = caller_field
        self["epsilon"] = epsilon
        self["max"] = None

    def dig(self, caller):
        if not self["max"] or self["max"] < caller[self["caller_field"]]:
            self["max"] = caller[self["caller_field"]]
            self.events["dig"](caller)

class RangeTrigger(abstract._Stage):
    """
        Low is inclusive, high exclusive
    """
    def __init__(self, *args, **kwargs):
        super(RangeTrigger, self).__init__(["dig"], *args, **kwargs)

    def _init(self, caller_field, low, high=None):
        self["caller_field"] = caller_field
        self["low"] = low
        self["high"] = high

    def dig(self, caller):
        if (
            (caller[self["caller_field"]] >= self["low"]) and self["high"] is None
            or (caller[self["caller_field"]] > self["low"] ) and ( caller[self["caller_field"]] < self["high"] )
        ) :
            self.events["dig"](caller)

class Print(abstract._Stage):
    """docstring for Print"""

    def __init__(self, *args, **kwargs):
        super(Print, self).__init__(None, *args, **kwargs)

    def _init(self, to_print):
        try:
            to_print.append
            self["message"] = to_print
        except :
            self["message"] = [to_print]

    def dig(self, caller):
        vals = list(self["message"])
        for i, val in enumerate(self["message"]) :
            vals[i] = abstract.freshest(val)
        print(*vals)

class PrettyPrintStore(abstract._Stage):
    """docstring for PrettyPrintStore"""
    def __init__(self, *args, **kwargs):
        super(PrettyPrintStore, self).__init__(None, *args, **kwargs)

    def _init(self, fields=None, prefix=""):
        if type(fields) is str :
            l_fields = [fields]
        else :
            l_fields = fields

        self.store["fields"] = l_fields
        self.store["prefix"] = prefix
        
    def dig(self, caller):
        store = utils.filter_dict(caller.store, self["fields"])
        store = utils.prefix_dict_keys(store, abstract.freshest( self["prefix"]) )
        
        res = [] 
        for k, v in store.items() :
            res.append("%s: %s" % (k, v))
        
        print("{\n", "\n  ".join(res), "\n}")

class CSVStreamSaver(abstract._Stage):
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
        tmp_store = utils.filter_dict(caller, self["select_fields"])
        
        store = {}
        for key, value in tmp_store.items():
            kkey = utils.apply_list_transform(key, self.fields_tranforms)
            vvalue = utils.apply_list_transform(value, self.values_tranforms)
            store["%s%s" % ( abstract.freshest(self["prefix"]), kkey)] = [vvalue]
        
        store["id"] = [self["line_number"]]

        df = pd.DataFrame.from_dict(store)
        df.to_csv(self.file, header=self["header"], index=False, **self.pandas_to_csv_kwargs)
        self["line_number"] += 1

    def close(self):
        self.file.close()

class MovingStats(abstract._Stage):
    """docstring for MovingStats"""
    def __init__(self, *args, **kwargs):
        super(MovingStats, self).__init__(["dig"], *args, **kwargs)
    
    def _init(self, caller_field, window_size=100):
        self["values"] = numpy.zeros(window_size)
        self["caller_field"] = caller_field
        self["counter"] = 0

    def dig(self, caller):
        self["values"][self["counter"] % len(self["values"]) ] = caller[self["caller_field"]]
        self["average"] = numpy.mean(self["values"])
        self["std"] = numpy.std(self["values"])
        self["min"] = numpy.min(self["values"])
        self["max"] = numpy.max(self["values"])
        self["counter"] += 1
        self.events["dig"](self)

class Stats(abstract._Stage):
    """docstring for Stats"""
    def __init__(self, *args, **kwargs):
        super(Stats, self).__init__(["dig"], *args, **kwargs)
    
    def _init(self, caller_field):
        self["values"] = []
        self["caller_field"] = caller_field
    
    def reset(self, caller) :
        def _do(caller) :
            self["values"] = []
            self["average"] = None
            self["std"] = None
            self["min"] = None
            self["max"] = None
        return _do

    def dig(self, caller):
        self["values"].append( caller[self["caller_field"]] )
        self["average"] = numpy.mean(self["values"])
        self["std"] = numpy.std(self["values"])
        self["min"] = numpy.min(self["values"])
        self["max"] = numpy.max(self["values"])
        self.events["dig"](self)

class StoreAggregator(abstract._Stage):
    """docstring for StoreAggregator"""
    def __init__(self, *args, **kwargs):
        super(StoreAggregator, self).__init__(["dig"], *args, **kwargs)
    
    def _init(self, select_fields, prefix):
        self["select_fields"] = select_fields
        self["prefix"] = prefix
        self["aggregate"] = {}

    def aggregate(self, caller):
        self["aggregate"].update(
            utils.filter_dict(caller, self["select_fields"])
        )

    def dig(self, caller):
        self.events["dig"](self)

class Break(abstract._Stage):
    """docstring for Break"""
    def __init__(self, *args, **kwargs):
        super(Break, self).__init__(["dig"], *args, **kwargs)
    
    def _init(self, reason=None, caller_breakpoint=None):
        self["reason"] = reason
        self["caller_breakpoint"] = caller_breakpoint

    def dig(self, caller):
        if self["caller_breakpoint"] is not None:
            raise exceptions.Break(self["caller_breakpoint"], self["reason"])
        else :
            raise exceptions.EndOfEverything(self["reason"])

class CSVWriter(abstract._Stage):
    """docstring for CSVWriter"""
    def __init__(self, *args, **kwargs):
        super(CSVWriter, self).__init__(["dig"], *args, **kwargs)

    def populate_current_line(self, dct) :
        self.resave = False
        if self["add_id"]:
            self["current_line"][self["legend"]["id"]] = str(len(self["lines"]))
    
        for k, v in dct.items():
            try :
                pos = self["legend"][k]
                self["current_line"][pos] = str(v)
            except KeyError:
                self["legend"][k] = len(self["legend"])
                self["current_line"].append(str(v))
                self.resave = True

    def add_dict_to_line(self, adct, prefix=None):
        def _do(caller) :
            dct = utils.prefix_dict_keys(adct, prefix)
            self.populate_current_line(dct)
        return _do

    def add_caller_to_line(self, fields, prefix, focus_caller=None):
        default_caller = focus_caller 
        def _do( caller ) :
            if default_caller is not None :
                cal = default_caller
            else :
                cal = caller
            dct = utils.filter_dict(cal.store, fields)
            dct = utils.prefix_dict_keys(dct, prefix)
            self.populate_current_line(dct)
        return _do

    def _init(self, filename, legend=None, separator="\t", line_separator="\n", na_handling="copy", add_id=True):
        from collections import OrderedDict
        
        self["lines"] = []
        self["legend"] = OrderedDict()
        self["add_id"] = add_id
        if self["add_id"] :
            self["legend"]["id"] = 0
        
        if legend is not None :
            for k in enumerate(legend):
                self["legend"][k] = len(self["legend"])
        
        self["na_handling"] = na_handling
        self["separator"] = separator
        self["line_separator"] = line_separator
        self["filename"] = filename
        self["current_line"] = None
        self.file = open(self["filename"], "w")
        self.resave = True
        self.formated_id = -1

    def _new_line(self) :
        if self["na_handling"] == "copy" and len(self["lines"]) > 0 :
            self["current_line"] = list(self.last_line)
        else :
            self["current_line"] = ["na"] * len(self["legend"])

    def open_line(self) :
        def _do(caller) :
            self._new_line()
        return _do

    def commit_line(self) :
        def _do(caller):
            self.last_line = self["current_line"]
            self["lines"].append(self["current_line"])
        return _do

    def save(self):
        import os
        def _do(caller):
            self.last_line = self["lines"][-1]
            for line_id in range(self.formated_id+1, len(self["lines"])) :
                self["lines"][line_id] = self["separator"].join(self["lines"][line_id])
                self.formated_id += 1

            if self.resave:
                res = self["separator"].join(self["legend"].keys()) + self["line_separator"] + self["line_separator"].join(self["lines"])           
                self.file.close()
                os.rename(self["filename"], self["filename"]+".bck")
                self.file = open(self["filename"], "w")
            else :
                res = self["line_separator"] + self["line_separator"].join(self["lines"])           

            self.file.write(res)
        return _do

    def dig(self, caller):
        self.events["dig"]()
