"""Digital scoreboard for Quizzy Mini: a red/blue score display in its own window,
meant to run fullscreen on a second monitor or a tablet propped up next to the robot.

Standalone check on a real display: `python -m scoreboard` opens the window and cycles
fake scores. Requires a Python build with Tkinter (the python.org installer has it; on
Homebrew macOS Python, `brew install python-tk`).

Module 1 usage -- feed it the same "score" / "reset" events the robot gets:

    board = Scoreboard()
    ...
    if event == "score":
        board.update(data["scores"])
    elif event == "reset":
        board.reset()
    ...
    board.close()

See INTEGRATION.md for the exact main.py wiring (one new line next to the existing
robot.set_score(...) calls).
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk

import config

log = logging.getLogger(__name__)

_POLL_MS = 100


class Scoreboard:
    """Owns its own Tk window and its own thread (Tkinter must be driven from a single
    thread, and main.py's game loop already owns the main thread). update()/reset()/close()
    are the only methods meant to be called from outside that thread; they are queue-based
    and safe to call from the game loop."""

    def __init__(self, *, title: str = "Quizzy Mini", win_score: int = config.WIN_SCORE):
        self.win_score = win_score
        self._queue: "queue.Queue[dict | None]" = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(title,), daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            log.warning("scoreboard window did not report ready in time; continuing anyway")

    # ------------------------------------------------------------------ main.py interface
    def update(self, scores: dict) -> None:
        """Game event `score`: scores is {"red": n, "blue": n}, the same payload passed to
        robot.set_score per team -- here it's taken whole."""
        self._queue.put({"red": scores.get("red", 0), "blue": scores.get("blue", 0)})

    def reset(self) -> None:
        """Game event `reset`: back to 0-0."""
        self.update({"red": 0, "blue": 0})

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2)

    # ------------------------------------------------------------------ Tk thread
    def _run(self, title: str) -> None:
        try:
            root = tk.Tk()
        except Exception:
            log.exception("could not open a scoreboard window (no display / no Tkinter?)")
            self._ready.set()
            return

        root.title(title)
        root.configure(bg="black")
        root.geometry("900x500")

        label_font = ("Helvetica", 36, "bold")
        score_font = ("Helvetica", 140, "bold")

        frame = tk.Frame(root, bg="black")
        frame.pack(expand=True, fill="both", padx=40, pady=40)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="RED", fg="#ff4d4d", bg="black", font=label_font).grid(row=0, column=0)
        tk.Label(frame, text="BLUE", fg="#4d79ff", bg="black", font=label_font).grid(row=0, column=1)
        red_var = tk.StringVar(value="0")
        blue_var = tk.StringVar(value="0")
        tk.Label(frame, textvariable=red_var, fg="#ff4d4d", bg="black", font=score_font).grid(row=1, column=0)
        tk.Label(frame, textvariable=blue_var, fg="#4d79ff", bg="black", font=score_font).grid(row=1, column=1)
        tk.Label(frame, text=f"first to {self.win_score}", fg="white", bg="black",
                 font=("Helvetica", 16)).grid(row=2, column=0, columnspan=2, pady=(10, 0))

        def on_close() -> None:
            self._queue.put(None)
            root.destroy()

        def poll() -> None:
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        root.destroy()
                        return
                    red_var.set(str(item["red"]))
                    blue_var.set(str(item["blue"]))
            except queue.Empty:
                pass
            root.after(_POLL_MS, poll)

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.after(_POLL_MS, poll)
        self._ready.set()
        try:
            root.mainloop()
        except Exception:
            log.exception("scoreboard window closed unexpectedly")


def _demo() -> None:
    """python -m scoreboard: open the board and cycle fake scores -- check readability on
    the actual monitor/tablet before game day."""
    import time

    logging.basicConfig(level=logging.INFO)
    board = Scoreboard()
    try:
        for red, blue in [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (5, 2)]:
            board.update({"red": red, "blue": blue})
            time.sleep(1.5)
        time.sleep(5)
    finally:
        board.close()


if __name__ == "__main__":
    _demo()
