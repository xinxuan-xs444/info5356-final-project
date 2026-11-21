"""Represents a quiz player."""
from dataclasses import dataclass, field

@dataclass
class Player:
    player_id: str
    name: str
    seat_id: int
    team: str
    answers_given: int = 0

@dataclass
class Team:
    color: str
    score: int = 0
    players: list[Player] = field(default_factory=list)

    def add_point(self):
        self.score += 1
