from . import exceptions

def freshest(value):
    if isinstance(value, StageFreshArgument) :
        return value()
    return value

class EventHandler(object):
    """docstring for EventHandler"""
    def __init__(self):
        super(EventHandler, self).__init__()
        self.events = []
    
    def add(self, event):
        self.events.append(event)

    def launch(self, caller):
        for evt in self.events:
            if isinstance(evt, _Stage) :
                evt._dig(caller)
            else:
                evt(caller)

    def __call__(self, *args, **kwargs):
        return self.launch(*args, **kwargs)

    def __getitem__(self, i):
        return self.events[i]

    def __setitem__(self, i, event):
        self.events[i] = events

    def __repr__(self):
        return "<%s (#%s), events: %s>" % (self.__class__.__name__, len(self.events), [ k for k in self.events ])

class StageFreshArgument(object):
    """docstring for StageFreshArgument"""
    def __init__(self, stage, key):
        super(StageFreshArgument, self).__init__()
        self.stage = stage
        self.key = key
    
    def __call__(self):
        return self.stage[self.key]

class _Stage(object):
    """docstring for _Stage"""
    def __init__(self, event_names=None, *args, **kwargs):
        super(_Stage, self).__init__()
        import inspect

        self.store = {}
        self.events = {}
        self._must_init = True
        self.focus_caller = None

        if event_names:
            if type(event_names) is str :
                e_names = [event_names]
            else :
                e_names = event_names
            self.events.update( { event_name: EventHandler() for event_name in e_names} )
        
        signature = inspect.signature(self._init)
        key_word_args = {}
        for i, item in enumerate(signature.parameters.items()):
            if item[0] not in ["args, "'kwargs'] :
                try:
                    key_word_args[item[0]] = args[i]
                except IndexError:
                    pass
                if item[1].default is not inspect.Parameter.empty:
                    key_word_args[item[0]] = item[1].default
        
        key_word_args.update(kwargs)
        if len(key_word_args) > 0 :
            try:
                self._init(**key_word_args)
                self._must_init = False
            except TypeError:
                self._must_init = True
        
    def focus(self, caller):
        """change the focus to point to a different caller than the one calling"""
        self.focus_caller = caller
        return self

    def setup(self, *args, **kwargs):
        self._init(*args, **kwargs)
        self._must_init = False
        return self

    def _init(self, *args , **kwargs):
        pass

    def _dig(self, caller):
        if self._must_init:
            raise AttributeError("Object  %s has not been initialized at creation. Try calling setup function" % (self)) 
        try:
            if self.focus_caller is not None :
                final_caller = self.focus_caller
            else :
                final_caller = caller
            return self.dig(final_caller)
        except exceptions.Break as e :
            if e.caller_breakpoint is self:
                pass
            else :
                raise e

    def dig(self, caller):
        raise NotImplemented("Must be implemented in child")

    def at(self, event_name, *stages):
        for stage in stages :
            try :
                self.events[event_name].add(stage)
            except KeyError as e :
                raise KeyError("%s has no event: %s. Has: %s" % (self, event_name, self.events.keys()))
        
        return self

    def get(self, key):
        """return as callable that will always return the current value of argment"""
        return StageFreshArgument(self, key)

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()
        
    def items(self):
        return self.store.items()
        
    def __call__(self, event_name, *stages):
        return self.at(event_name, *stages)

    def __getitem__(self, k):
        try :
            # return self.get(k)
            return self.store[k]
        except KeyError as e :
            raise KeyError("%s has no attribute: %s. Has: %s" % (self, k, self.keys()))

    def __setitem__(self, k, v):
        self.store[k] = v

    def __repr__(self):
        return "<%s, events: %s>" % (self.__class__.__name__, [ k for k in self.events.keys() ])
