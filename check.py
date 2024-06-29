import tkinter as tk
import numpy as np
from mlinterface.gui.backend.lookback import Lookback
from mlinterface.gui.components.variables import GenericVar


app = tk.Tk()

print("Out", GenericVar(value=111).get())
print("Out", GenericVar(value=1.2).get())
print("Out", GenericVar(value="a").get())


a = GenericVar(value=111)
b = GenericVar(value=1.2)
c = GenericVar(value="a")

tk.Entry(app, textvariable=a).pack()
tk.Entry(app, textvariable=b).pack()
tk.Entry(app, textvariable=c).pack()

tk.Button(app, text="Submit", command=lambda: print(a.get(), b.get(), c.get())).pack()

app.mainloop()

# X = np.random.rand(100, 3)
# y = np.random.rand(100)
#
# lookback = Lookback(5, 4, 2)
# X, y = lookback.get_lookback(X, y)
#
# X_test = np.random.rand(5, 3)
# y_test = np.random.rand(5)
#
#
# print("Last:", lookback.last)
# print("X_test:", X_test[-2])
# print("Appended:", lookback.append_lookback(X_test[-2]))
#
# lookback.update_last(3)
#
# print("Updated with value 3")
# print("X_test:", X_test[-1])
# print("Appended:", lookback.append_lookback(X_test[-1]))
#
#
#
#
# X = np.random.rand(10, 3)
# y = np.random.rand(10)
#
# print("X:", X)
#
# lookback = Lookback(0, 0, 0)
# X, y = lookback.get_lookback(X, y)
#
# print("X:", X)
#
