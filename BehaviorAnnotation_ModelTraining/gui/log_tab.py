# content of gui/log_tab.py

import tkinter as tk
from tkinter import ttk, scrolledtext

class ConsoleRedirector:
    """A class to redirect stdout and stderr to a Tkinter text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.config(state=tk.NORMAL) # Ensure it's writable

    def write(self, message):
        # The master might be destroyed when the app is closing.
        if self.text_widget.master.winfo_exists():
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END) # Auto-scroll
            self.text_widget.update_idletasks()

    def flush(self):
        # This is needed for compatibility with the file-like object interface.
        pass

def create_log_tab(app):
    """
    Creates the 'Log' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    # Create a new tab for logging
    app.log_tab = ttk.Frame(app.notebook, padding="10")
    app.notebook.add(app.log_tab, text='Log')

    # Create a container frame
    log_container = ttk.Frame(app.log_tab)
    log_container.pack(expand=True, fill=tk.BOTH)
    log_container.columnconfigure(0, weight=1)
    log_container.rowconfigure(0, weight=1)

    # Add a scrolled text widget for the log output
    app.log_text_widget = scrolledtext.ScrolledText(
        log_container, 
        wrap=tk.WORD, 
        state=tk.DISABLED, # Start as read-only
        bg="#2b2b2b", 
        fg="#d3d3d3",
        font=("Consolas", 9)
    )
    app.log_text_widget.grid(row=0, column=0, sticky="nsew")

    # Add a button to clear the log
    button_frame = ttk.Frame(log_container)
    button_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

    def clear_log_text():
        """Clears the content of the log text widget."""
        app.log_text_widget.config(state=tk.NORMAL)
        app.log_text_widget.delete('1.0', tk.END)
        app.log_text_widget.config(state=tk.DISABLED)

    # Attach the clear function to the app instance
    app.clear_log = clear_log_text

    clear_button = ttk.Button(button_frame, text="Clear Log", command=app.clear_log)
    clear_button.pack(side=tk.RIGHT)