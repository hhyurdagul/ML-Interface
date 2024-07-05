from tkinter import NORMAL, DISABLED
from tkinter.ttk import Entry
from tkinter import Variable


class GenericIntVar(Variable):
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
        value = self._tk.globalgetvar(self._name)  # type: ignore
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def reset(self):
        self.set(self.default)


class GenericFloatVar(Variable):
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
        value = self._tk.globalgetvar(self._name)  # type: ignore
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    def reset(self):
        self.set(self.default)


def change_state(
    state: int,
    entries: list[Entry],
    variables: list[GenericIntVar | GenericFloatVar] | None = None,
) -> None:
    for entry in entries:
        entry["state"] = NORMAL if state else DISABLED

    if variables is not None and not state:
        for variable in variables:
            variable.reset()
