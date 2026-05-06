import subprocess
import threading
import os
import sys
import signal
import shutil
import re
from tkinter import messagebox
import traceback


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class ProcessManager:
    def __init__(self, app):
        self.app = app

    def start_process(self, command_builder_func, process_attr, on_finish_callback=None):
        if getattr(self.app, process_attr, None) and getattr(self.app, process_attr).poll() is None:
            self.app.log_message(f"A {process_attr} is already running.", "WARNING")
            messagebox.showwarning("In Progress", "A process is already running.", parent=self.app.root)
            return
        
        try:
            cmd, err = command_builder_func(self.app)
            if err:
                self.app.log_message(f"Command Builder Error: {err}", "ERROR")
                messagebox.showerror("Input Validation Error", err, parent=self.app.root)
                return
            
            self._start_and_stream_process(cmd, process_attr, on_finish_callback)
        except Exception as e:
            self.app.log_message(f"Process Start Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Process Error", f"Failed to start process: {e}", parent=self.app.root)

    def _start_and_stream_process(self, command, process_attribute, on_finish_callback=None):
        self.app.log_message(f"--- Starting Command ---\n{' '.join(command)}\n------------------------", "INFO")
        try:
            executable = command[0]
            if shutil.which(executable) is None and not os.path.isabs(executable):
                msg = (
                    f"Executable '{executable}' was not found on PATH. "
                    "Ensure the Ultralytics CLI is installed (pip install ultralytics) "
                    "and accessible before running this action."
                )
                self.app.log_message(msg, "ERROR")
                messagebox.showerror("Command Not Found", msg, parent=self.app.root)
                if on_finish_callback:
                    self.app.root.after(0, on_finish_callback)
                return

            popen_kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT, 'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'bufsize': 1}
            if sys.platform != "win32":
                popen_kwargs['preexec_fn'] = os.setsid
            process = subprocess.Popen(command, **popen_kwargs)
            setattr(self.app, process_attribute, process)
            threading.Thread(target=self._stream_process_output, args=(process, on_finish_callback), daemon=True).start()
        except Exception as e:
            self.app.log_message(f"ERROR starting process: {e}\n{traceback.format_exc()}", "ERROR")
            if on_finish_callback:
                self.app.root.after(0, on_finish_callback)

    def _stream_process_output(self, process, on_finish_callback):
        for line in iter(process.stdout.readline, ''):
            if line:
                # Preserve leading spaces for CLI table alignment (e.g., Ultralytics val metrics).
                cleaned = _ANSI_ESCAPE_RE.sub("", line.rstrip("\r\n"))
                self.app.log_queue.put(cleaned)
        rc = process.wait()
        self.app.log_queue.put(f"--- Process finished with exit code {rc} ---")
        if on_finish_callback:
            self.app.root.after(0, on_finish_callback)

    def terminate_process(self, process_attribute):
        process = getattr(self.app, process_attribute, None)
        if process and process.poll() is None:
            self.app.log_message(f"Stopping {process_attribute} (PID: {process.pid})", "INFO")
            try:
                if sys.platform == "win32":
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], check=True, capture_output=True)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=3)
            except Exception:
                process.kill()
            finally:
                setattr(self.app, process_attribute, None)
                self.app.log_message(f"Terminated {process_attribute}", "INFO")
