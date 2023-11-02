from tkinter import messagebox


def popupmsg(msg: str) -> bool:
    messagebox.showinfo("!", msg)
    return False
