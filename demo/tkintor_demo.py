from tkinter import *

root = Tk()

w = Canvas(
    root,
    width=200,
    height=200,
    background="white"
)
w.pack()

w.create_rectangle(
    50, 50,
    100, 100,
    fill='blue'
)

mainloop()
