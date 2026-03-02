# 说白了，这就是一个 GUI

import tkinter as tk
import os

def open_terminal():
    os.system('start powershell')  # Windows
    # os.system('gnome-terminal')  # Linux
    # os.system('open -a Terminal')  # macOS

f5k_root = tk.Tk()  # F*****g
f5k_root.title("《GUI》")
f5k_root.geometry("750x300")

that_d2n_text_used_to_explain_that_this_is_a_GUI = """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
THIS IS A GUI TOO.
YES, IT REALLY IS.

Open the ALL-POWERFUL PowerShell here
and CONQUER whatever you want.

WHY does a Reinforcement Learning project
need a COMPLEX GUI?
IT DOESN'T.

THIS. IS. ENOUGH.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""  # AI 写的
tk.Label(f5k_root, text=that_d2n_text_used_to_explain_that_this_is_a_GUI, font=("Consolas", 8)).pack(expand=True, fill="both")
tk.Button(f5k_root, text="Open PowerShell", command=open_terminal, font=("Consolas", 64)).pack(expand=True, fill="both")

f5k_root.mainloop()