"""Keyboard input and operator overrides for the active quiz."""

import logging
import math
from queue import Empty, Queue
from threading import Thread

import config
from game.game_controller import GameController, GameState

logger = logging.getLogger(__name__)
HELP = (
    "mode name|team | player SEAT red|blue NAME | start | buzz SEAT | "
    "answer NUMBER | correct | incorrect | skip | status | pause | resume | reset | quit"
)


class OperatorConsole:
    def __init__(self, game: GameController, *, on_tick=None) -> None:
        self.game = game
        self.on_tick = on_tick

    def run(self) -> None:
        commands = Queue()

        def read_commands() -> None:
            try:
                while True:
                    command = input()
                    commands.put(command)
                    if command.strip().lower() == "quit":
                        return
            except EOFError:
                commands.put("quit")

        print("Quizzy Mini's Math Showdown")
        print(HELP)
        Thread(target=read_commands, daemon=True).start()
        while True:
            self.game.update()
            if self.on_tick is not None:
                self.on_tick()
            try:
                command = commands.get(timeout=0.05)
            except Empty:
                continue
            if not self.execute(command):
                return

    def execute(self, command: str) -> bool:
        action, _, argument = command.strip().partition(" ")
        action = action.lower()
        argument = argument.strip()
        game = self.game
        accepted = True
        try:
            if action == "quit":
                return False
            if action == "mode":
                if game.state is not GameState.INTRODUCE or argument not in (
                    config.ADDRESSING_NAME, config.ADDRESSING_TEAM
                ):
                    raise ValueError("Choose mode name or mode team before start")
                game.addressing_mode = argument
            elif action == "player":
                seat, team, name = argument.split(maxsplit=2)
                game.add_player(name, int(seat), team)
            elif action == "start":
                game.start()
            elif action == "buzz":
                accepted = game.buzz(int(argument), override=True)
            elif action == "answer":
                answer = float(argument)
                if not math.isfinite(answer):
                    raise ValueError("Enter a finite number")
                accepted = game.answer(answer)
            elif action in ("correct", "incorrect"):
                accepted = game.answer(judgment=action == "correct")
            elif action in ("skip", "pause", "resume", "reset"):
                accepted = getattr(game, action)() is not False
            elif action == "status":
                now = game.clock() if game.paused_at is None else game.paused_at
                print(f"State: {game.state.name}{' (paused)' if game.paused_at is not None else ''}")
                for label, deadline in (("Game", game.game_deadline), ("Phase", game.phase_deadline)):
                    if deadline is not None:
                        print(f"{label}: {max(0, deadline - now):.1f}s remaining")
                seats = game.eligible_seats()
                if seats:
                    print(f"Eligible seats: {', '.join(map(str, seats))}")
                game.publish_scores()
            elif action:
                print(HELP)
        except ValueError as error:
            print(error)
            return True

        if not accepted:
            print("That command is unavailable in the current state.")
        elif action in ("buzz", "answer", "correct", "incorrect", "skip", "pause", "resume", "reset"):
            logger.info("Manual command: %s %s", action, argument)
        return True
