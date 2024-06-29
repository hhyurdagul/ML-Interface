from tkinter import Variable

class GenericVar(Variable):
    """Value holder for generic variables."""
    _default = ""

    def __init__(self, master=None, value=None, name=None):
        """Construct an integer variable.

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to 0)
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        self.default = value
        Variable.__init__(self, master, value, name)

    def get(self):
        """Return the value of the variable as an integer."""
        value = self._tk.globalgetvar(self._name)
        if value == "":
            return 0
        else:
            try:
                return int(value)
            except ValueError:
                return float(value)

    def reset(self):
        self.set(self.default)
