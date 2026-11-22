"""Run the two-team quiz without depending on terminal or hardware I/O."""

import math
import time
from dataclasses import dataclass
from enum import Enum, auto

import config
from game.player import Player, Team
from game.question_bank import QuestionBank


class GameState(Enum):
    INTRODUCE = auto()
    ASK = auto()
    THINK = auto()
    BUZZ_OPEN = auto()
    CALL = auto()
    JUDGE = auto()
    STEAL = auto()
    HINT_THINK = auto()
    TEACH = auto()
    SCORE = auto()
    END = auto()


class Attempt(Enum):
    FIRST = auto()
    STEAL = auto()
    HINT = auto()
    MINI = auto()


@dataclass
class Round:
    question: dict
    attempt: Attempt = Attempt.FIRST
    first_team: str | None = None
    player: Player | None = None


class GameController:
    def __init__(self, question_bank=None, *, clock=time.monotonic, emit=None):
        self.question_bank = question_bank if question_bank is not None else QuestionBank()
        self.clock = clock
        self.emit = emit if emit is not None else lambda event, **payload: None
        self.addressing_mode = config.ADDRESSING_NAME
        self.reset()

    def reset(self) -> None:
        self.state = GameState.INTRODUCE
        self.teams = {color: Team(color) for color in ("red", "blue")}
        self.players: dict[int, Player] = {}
        self.round: Round | None = None
        self.game_deadline: float | None = None
        self.phase_deadline: float | None = None
        self.paused_at: float | None = None
        self.sudden_death = False
        self.winner: str | None = None
        self.question_bank.reset()
        self.emit("reset")
        self.publish_scores()

    def add_player(self, name: str, seat: int, color: str) -> None:
        if self.state is not GameState.INTRODUCE:
            raise ValueError("Register players before starting")
        if not 1 <= seat <= config.MAX_PLAYERS or seat in self.players:
            raise ValueError(f"Choose an unused seat from 1 to {config.MAX_PLAYERS}")
        color = color.lower()
        if color not in self.teams or not name.strip():
            raise ValueError("Provide a name and a team color (red or blue)")
        team = self.teams[color]
        if len(team.players) >= (config.MAX_PLAYERS + 1) // 2:
            raise ValueError("That team is full")
        player = Player(f"P{seat:02}", name.strip(), seat, color)
        self.players[seat] = player
        team.players.append(player)

    def start(self) -> None:
        if self.state is not GameState.INTRODUCE:
            raise ValueError("Reset before starting another game")
        if not config.MIN_PLAYERS <= len(self.players) <= config.MAX_PLAYERS:
            raise ValueError(f"Register {config.MIN_PLAYERS} or {config.MAX_PLAYERS} players")
        if any(len(team.players) < config.MIN_PLAYERS // 2 for team in self.teams.values()):
            raise ValueError("Register at least two players on each team")
        if self.addressing_mode not in (config.ADDRESSING_NAME, config.ADDRESSING_TEAM):
            raise ValueError("Choose name or team addressing")
        self.game_deadline = self.clock() + config.GAME_DURATION
        self.publish_scores()
        self.next_question()

    def transition(self, state: GameState, duration: float | None = None) -> None:
        self.state = state
        self.phase_deadline = None if duration is None else self.clock() + duration
        self.emit("state", state=state.name)

    def say(self, text: str) -> None:
        self.emit("say", text=text)

    def publish_scores(self) -> None:
        self.emit("score", scores={color: team.score for color, team in self.teams.items()})

    def get_player_address(self, player: Player) -> str:
        if self.addressing_mode == config.ADDRESSING_NAME:
            return player.name.split()[0]
        return f"{player.team.title()} team"

    def leading_team(self) -> str | None:
        red, blue = self.teams["red"].score, self.teams["blue"].score
        if red == blue:
            return None
        return "red" if red > blue else "blue"

    def next_question(self) -> None:
        question = self.question_bank.next_question()
        if question is None:
            self.finish(self.leading_team())
            return
        self.round = Round(question)
        self.transition(GameState.ASK)
        self.emit("question", question=question["question"], difficulty=question["difficulty"])
        self.say(question["question"])
        self.transition(GameState.THINK, config.THINK_DURATION)
        self.say(f"Discuss for {config.THINK_DURATION} seconds.")

    def open_buzz(self, state: GameState, prompt: str = "Buzz open!") -> None:
        self.transition(state, config.HINT_BUZZ_DURATION)
        self.emit("buzz_open", seats=self.eligible_seats())
        self.say(prompt)

    def eligible_seats(self) -> tuple[int, ...]:
        if self.state not in (GameState.BUZZ_OPEN, GameState.STEAL, GameState.TEACH):
            return ()
        if self.state is GameState.STEAL:
            return tuple(sorted(
                seat for seat, player in self.players.items()
                if player.team != self.round.first_team
            ))
        if self.state is GameState.TEACH and self.round.attempt is Attempt.HINT:
            return tuple(sorted(self.players))
        eligible = []
        for team in self.teams.values():
            fewest = min(player.answers_given for player in team.players)
            eligible.extend(player.seat_id for player in team.players
                            if player.answers_given == fewest)
        return tuple(sorted(eligible))

    def buzz(self, seat: int, *, override: bool = False) -> bool:
        if self.paused_at is not None:
            return False
        if self.state not in (GameState.BUZZ_OPEN, GameState.STEAL, GameState.TEACH):
            return False
        window = (self.state, self.phase_deadline)
        self.update()
        if (self.state, self.phase_deadline) != window:
            return False
        player = self.players.get(seat)
        if player is None:
            return False
        if (self.state is GameState.STEAL or not override) and seat not in self.eligible_seats():
            return False
        self.round.player = player
        if self.round.attempt is Attempt.FIRST:
            self.round.first_team = player.team
        self.transition(GameState.CALL)
        self.emit("look_at_seat", seat=seat)
        self.say(f"{self.get_player_address(player)}, what's your answer?")
        self.transition(GameState.JUDGE, config.ANSWER_DURATION)
        self.emit("answer_open", seat=seat, timeout=config.ANSWER_DURATION)
        return True

    def check_answer(self, answer) -> bool:
        field = "mini_answer" if self.round.attempt is Attempt.MINI else "answer"
        if isinstance(answer, bool):
            return False
        try:
            return math.isclose(float(answer), self.round.question[field], rel_tol=0, abs_tol=1e-6)
        except (TypeError, ValueError, OverflowError):
            return False

    def answer(self, value=None, *, judgment: bool | None = None) -> bool:
        self.update()
        if self.paused_at is not None or self.state is not GameState.JUDGE:
            return False
        self.round.player.answers_given += 1
        self.judge(self.check_answer(value) if judgment is None else judgment)
        return True

    def judge(self, correct: bool, *, timed_out: bool = False) -> None:
        attempt = self.round.attempt
        feedback = "Time is up." if timed_out else ("Correct!" if correct else "Incorrect.")
        if attempt is Attempt.FIRST and not correct:
            self.round.attempt = Attempt.STEAL
            other = "blue" if self.round.first_team == "red" else "red"
            self.open_buzz(GameState.STEAL, f"{feedback} {other.title()} team, buzz now to steal!")
            return
        self.say(feedback)
        if attempt is Attempt.MINI:
            if not correct:
                self.say(f"The practice answer is {self.round.question['mini_answer']}.")
            self.next_question()
        elif correct:
            self.transition(GameState.SCORE)
            color = self.round.player.team
            self.teams[color].add_point()
            self.publish_scores()
            if self.sudden_death or self.teams[color].score >= config.WIN_SCORE:
                self.finish(color)
            else:
                self.next_question()
        elif attempt is Attempt.STEAL:
            self.start_teaching()
        else:
            self.show_solution()

    def start_teaching(self) -> None:
        self.round.attempt = Attempt.HINT
        self.say(f"Hint: {self.round.question['hint']}")
        self.say(f"Take {config.HINT_THINK_DURATION} seconds to work it out.")
        self.transition(GameState.HINT_THINK, config.HINT_THINK_DURATION)

    def show_solution(self) -> None:
        self.round.attempt = Attempt.MINI
        for step in self.round.question["solution"]:
            self.say(step)
        self.say(self.round.question["mini_question"])
        self.open_buzz(GameState.TEACH)

    def update(self) -> None:
        if self.paused_at is not None or self.state in (GameState.INTRODUCE, GameState.END):
            return
        now = self.clock()
        if self.game_deadline is not None and now >= self.game_deadline:
            leader = self.leading_team()
            if leader is not None:
                self.finish(leader)
            else:
                self.game_deadline = None
                self.sudden_death = True
                self.say("Tie game! Sudden death: the next scoring answer wins.")
                self.next_question()
            return
        if self.phase_deadline is None or now < self.phase_deadline:
            return
        if self.state is GameState.THINK:
            self.open_buzz(GameState.BUZZ_OPEN)
        elif self.state is GameState.HINT_THINK:
            self.open_buzz(GameState.TEACH, "Buzz open to both teams!")
        elif self.state in (GameState.BUZZ_OPEN, GameState.STEAL):
            self.start_teaching()
        elif self.state is GameState.JUDGE:
            self.judge(False, timed_out=True)
        elif self.round.attempt is Attempt.HINT:
            self.show_solution()
        else:
            self.say(f"The practice answer is {self.round.question['mini_answer']}.")
            self.next_question()

    def skip(self) -> bool:
        if self.state in (GameState.INTRODUCE, GameState.END) or self.paused_at is not None:
            return False
        self.next_question()
        return True

    def pause(self) -> bool:
        self.update()
        if self.state in (GameState.INTRODUCE, GameState.END) or self.paused_at is not None:
            return False
        self.paused_at = self.clock()
        return True

    def resume(self) -> bool:
        if self.paused_at is None:
            return False
        elapsed = self.clock() - self.paused_at
        if self.game_deadline is not None:
            self.game_deadline += elapsed
        if self.phase_deadline is not None:
            self.phase_deadline += elapsed
        self.paused_at = None
        return True

    def finish(self, winner: str | None) -> None:
        self.winner = winner
        self.game_deadline = None
        self.transition(GameState.END)
        self.say(f"{winner.title()} team wins!" if winner else "The game ends in a draw.")
