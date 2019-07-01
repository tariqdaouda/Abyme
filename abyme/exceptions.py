class EndOfEverything(Exception):
    """docstring for EndOfEverything"""
    pass

class Break(Exception):
    """docstring for Break"""
    def __init__(self, caller_breakpoint, *args, **kwargs):
        super(Break, self).__init__(*args, **kwargs)
        self.caller_breakpoint = caller_breakpoint        