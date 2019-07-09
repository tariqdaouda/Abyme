from . import abstract
from . import utils
from . import exceptions

import numpy
import time
import os

class Ground(abstract._Stage):
    """docstring for Ground"""
    def __init__(self, *args, **kwargs):
        super(Ground, self).__init__(["dig", "final"])
    
    def dig(self, caller=None):
        try:
            self.events["dig"](self)        
        except exceptions.EndOfEverything as e:
            print("Ended because:", e)
            self.events["final"](self)        

class Mkdir(abstract._Stage):
    """docstring for Mkdir"""
    def __init__(self, *args, **kwargs):
        super(Mkdir, self).__init__(["start", "folder_created", "entered_folder", "end"])
    
    def _init(self, folder, enter_folder=False, add_date=False):
        self["folder"] = folder
        if add_date :
            self["folder"] = time.ctime().replace(" ", "-") + "_" + folder

        i = 0
        fol = self["folder"]
        while os.path.isdir(self["folder"]) :
            self["folder"] = fol + "_%d" % i 
            i += 1

        self["enter_folder"] = enter_folder
  
    def dig(self, caller):
        self.events["start"](self)
        os.mkdir(self["folder"])
        self.events["folder_created"](self)
        if self["enter_folder"] :
            os.chdir(self["folder"])
            self.events["entered_folder"](self)
    
        self.events["end"](self)

class PerformanceProfiler(abstract._Stage):
    """Profiles the performances"""
    def __init__(self, *args, **kwargs):
        super(PerformanceProfiler, self).__init__(
            [
                "dig",
                "before_record_start",
                "after_record_start",
                "before_record_end",
                "after_record_end",
            ])
    
    def _init(self, to_int=True):
        self["to_int"] = to_int
        self["start_time"] = None
        self["end_time"] = None
        self["elapsed_time"] = None
        
    def record_start(self, caller):
        self.events["before_record_end"](self)
        self["start_time"] = time.perf_counter()
        if self["to_int"] :
            self["start_time"] = int(self["start_time"])
        self.events["after_record_start"](self)
        
    def record_end(self, caller):
        self.events["before_record_end"](self)
        self["end_time"] = time.perf_counter()
        if self["to_int"] :
            self["end_time"] = int(self["end_time"])
        self["elapsed_time"] = self["end_time"] - self["start_time"]
        self.events["after_record_end"](self)
        
    def dig(self, caller):
        self.events["dig"](self)
        
class Run(abstract._Stage):
    """docstring for Run"""
    def __init__(self, *args, **kwargs):
        super(Run, self).__init__(["start", "end"])
    
    def _init(self, capture_result, something_callable, *callable_args, **callable_kwargs):
        self["callable"] = something_callable
        self["callable_args"] = callable_args
        self["callable_kwargs"] = callable_kwargs
        self["capture_result"] = capture_result
        self["callable_result"] = None

    def dig(self, caller):
        self.events["start"](self)
        res = self["callable"](*self["callable_args"], **self["callable_kwargs"])
        if self["capture_result"] :
            self["callable_result"] = res
        self.events["end"](self)
        
class Chdir(abstract._Stage):
    """docstring for Chdir"""
    def __init__(self, *args, **kwargs):
        super(Chdir, self).__init__(["dig"])
    
    def _init(self, folder):
        self["folder"] = folder

    def dig(self, caller):
        os.chdir(self["folder"])
        self.events["dig"](self)

class IterrationLooper(abstract._Stage):
    """docstring for IterrationLooper"""
    def __init__(self, *args, **kwargs):
        super(IterrationLooper, self).__init__(["start", "iteration_start", "iteration_end", "end"], *args, **kwargs)
    
    def _init(self, nb_iterations):
        self["nb_iterations"] = nb_iterations
        self["counter"] = 0
        self["advancement"] = self["counter"]/self["nb_iterations"] 
        self["perc_advancement"] = (self["counter"]/self["nb_iterations"]) * 100

    def dig(self, caller):    
        self.events["start"](self)
        max_it = self["nb_iterations"]
        while max_it != 0 :
            self.events["iteration_start"](self)
            self["counter"] += 1
            self.events["iteration_end"](self)
            max_it -= 1
            self["advancement"] = self["counter"]/self["nb_iterations"] 
            self["perc_advancement"] = (self["counter"]/self["nb_iterations"]) * 100
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
        import types
        val = abstract.call_if_callable(self["condition"])
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
        self["values"] = []
        self["average"] = None
        self["std"] = None
        self["min"] = None
        self["max"] = None
        
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

    def _init(self, filename, fieldnames, na_handling="copy", add_id=True, csv_writer_kwargs = {}):
        if not csv_writer_kwargs :
            csv_writer_kwargs = {}
        self.csv_writer_kwargs = csv_writer_kwargs

        self["filename"] = filename
        self["na_handling"] = na_handling
        self["fieldnames"] = fieldnames
        self["add_id"] = add_id
        if self["add_id"] :
            self["fieldnames"].insert(0, "id")
        self["counter"] = 0

    def open(self, caller):
        import csv

        self.last_line = None
        self.file = open(self["filename"], "w", newline="")
        self.writer = csv.DictWriter(self.file, self["fieldnames"], **self.csv_writer_kwargs)
        self.writer.writeheader()
        self._newline()

    def _newline(self) :
        if self["na_handling"] == "copy" and self.last_line is not None :
            self["current_line"] = dict(self.last_line)
        else :
            self["current_line"] = { k: "na" for k in self["fieldnames"]}

        if self["add_id"]:
            self["current_line"]["id"] = self["counter"]

    def open_line(self) :
        def _do(caller) :
            self._newline()
        return _do

    def commit_line(self) :
        def _do(caller):
            self.writer.writerow(self["current_line"])
            self.last_line = self["current_line"]        
            self["counter"] += 1
        return _do

    def populate_current_line(self, dct) :
        for k in dct.keys():
            self["current_line"][k] = dct[k]

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

    def dig(self, caller):
        self.events["dig"]()
