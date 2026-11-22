"""Digital scoreboard for Quizzy Mini: a red/blue score display in its own window,
meant to run fullscreen on a second monitor or a tablet propped up next to the robot.

Standalone check on a real display: `python -m scoreboard` opens the window and cycles
fake scores. Requires a Python build with Tkinter (the python.org installer has it; on
Homebrew macOS Python, `brew install python-tk`; conda's default Python also has it).

macOS requires all GUI windows to be created and driven from the main thread, so unlike
RobotMotion this does NOT run its own background thread. Construct it on the same thread
as everything else (main.py's main thread) and call pump() once per loop tick from
wherever that thread already polls -- OperatorConsole.run()'s existing ~50ms loop is the
natural place. See INTEGRATION.md.

Module 1 usage -- feed it the same "score" / "reset" events the robot gets, and pump it
alongside the existing game.update() call:

    board = Scoreboard()
    ...
    if event == "score":
        board.update(data["scores"])
    elif event == "reset":
        board.reset()
    ...
    # once per loop tick, next to self.game.update():
    board.pump()
    ...
    board.close()
"""

from __future__ import annotations

import logging
import tkinter as tk

import config

log = logging.getLogger(__name__)


class Scoreboard:
    """Must be constructed on, and pumped from, a single thread (macOS: the main thread).
    update()/reset()/pump()/close() are all meant to be called from that same thread --
    there is no internal locking, unlike RobotMotion, because there is no second thread."""

    def __init__(self, *, title: str = "Quizzy Mini", win_score: int = config.WIN_SCORE):
        self.win_score = win_score
        self.root: tk.Tk | None = None
        self._closed = False
        try:
            self.root = tk.Tk()
        except Exception:
            log.exception("could not open a scoreboard window (no display / no Tkinter?)")
            self._closed = True
            return

        self.root.title(title)
        self.root.configure(bg="black")
        self.root.geometry("900x500")

        label_font = ("Helvetica", 36, "bold")
        score_font = ("Helvetica", 140, "bold")

        frame = tk.Frame(self.root, bg="black")
        frame.pack(expand=True, fill="both", padx=40, pady=40)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="RED", fg="#ff4d4d", bg="black", font=label_font).grid(row=0, column=0)
        tk.Label(frame, text="BLUE", fg="#4d79ff", bg="black", font=label_font).grid(row=0, column=1)
        self._red = tk.StringVar(value="0")
        self._blue = tk.StringVar(value="0")
        tk.Label(frame, textvariable=self._red, fg="#ff4d4d", bg="black", font=score_font).grid(row=1, column=0)
        tk.Label(frame, textvariable=self._blue, fg="#4d79ff", bg="black", font=score_font).grid(row=1, column=1)
        tk.Label(frame, text=f"first to {self.win_score}", fg="white", bg="black",
                 font=("Helvetica", 16)).grid(row=2, column=0, columnspan=2, pady=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_window_closed)

    # ------------------------------------------------------------------ main.py interface
    def update(self, scores: dict) -> None:
        """Game event `score`: scores is {"red": n, "blue": n}, the same payload passed to
        robot.set_score per team -- here it's taken whole. No-op once the window is closed."""
        if self._closed or self.root is None:
            return
        self._red.set(str(scores.get("red", 0)))
        self._blue.set(str(scores.get("blue", 0)))

    def reset(self) -> None:
        """Game event `reset`: back to 0-0."""
        self.update({"red": 0, "blue": 0})

    def pump(self) -> None:
        """Call once per loop tick, from the same thread that constructed this Scoreboard,
        to process window events and repaint -- there is no mainloop() here on purpose,
        since main.py's own loop already owns the thread. A no-op once the window is closed
        (by the user clicking its close button, or after close() is called)."""
        if self._closed or self.root is None:
            return
        try:
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self) -> None:
        if self.root is not None and not self._closed:
            try:
                self.root.destroy()
            except Exception:
                pass
        self._closed = True

    def _on_window_closed(self) -> None:
        self._closed = True
        if self.root is not None:
            self.root.destroy()


def _demo() -> None:
    """python -m scoreboard: open the board and cycle fake scores -- check readability on
    the actual monitor/tablet before game day. Pumps itself in a simple loop since there's
    no main.py here to do it."""
    import time

    logging.basicConfig(level=logging.INFO)
    board = Scoreboard()
    try:
        for red, blue in [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (5, 2)]:
            board.update({"red": red, "blue": blue})
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and not board._closed:
                board.pump()
                time.sleep(0.03)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not board._closed:
            board.pump()
            time.sleep(0.03)
    finally:
        board.close()


if __name__ == "__main__":
    _demo()
