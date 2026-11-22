"""Application configuration for Quizzy Mini."""
# All code files should read parameters from config.py.

GAME_DURATION = 300      # 5 min
THINK_DURATION = 20      # 20 sec to think after the question is read
ANSWER_DURATION = 10     # 10 sec to answer after raising hand
HINT_THINK_DURATION = 5  # 5 sec to think after the hint
HINT_BUZZ_DURATION = 15  # 15 sec waiting for a buzz

WIN_SCORE = 5
MAX_PLAYERS = 4
MIN_PLAYERS = 3

QUESTION_COUNT = 20



# Experimental conditions
ADDRESSING_NAME = "name"
ADDRESSING_TEAM = "team"

# --- Module 4: Robot Motion & Scoreboard ---
# All angles in degrees. Calibrate with `python -m calibration seats` / `python -m calibration antennas`
# before a real session; these are placeholders until then.

# Seat -> (yaw, pitch) for the head to look at that seat. Direction (which physical side is
# Seat 1) must match game logic's seat numbering and vision.SeatMap(seat_ids=...) -- still
# "to agree as a team" per vision/INTEGRATION.md, so treat these as provisional.
SEAT_HEAD_ANGLES = {
    1: (-30.0, -5.0),
    2: (-10.0, -5.0),
    3: (10.0, -5.0),
    4: (30.0, -5.0),
}
HEAD_NEUTRAL_ANGLE = (0.0, -5.0)   # also the pose vision's camera assumes when a buzz window opens
HEAD_MOVE_DURATION = 0.6           # seconds for a look_at_seat / look_neutral move

# Hard safety clamps, independent of calibration above -- never send the head or antennas
# past these regardless of what a bad calibration value says.
HEAD_SAFE_YAW_RANGE = (-45.0, 45.0)
HEAD_SAFE_PITCH_RANGE = (-20.0, 20.0)
ANTENNA_SAFE_RANGE = (-90.0, 150.0)
MAX_MOVE_SPEED_DEG_S = 180.0

# Antenna angle for each score 0..WIN_SCORE (index = score), one needle per team.
ANTENNA_SCORE_ANGLES = (-60.0, -24.0, 12.0, 48.0, 84.0, 120.0)
ANTENNA_WIGGLE_DEGREES = 20.0
ANTENNA_WIGGLE_CYCLES = 2

IDLE_ANIMATION_INTERVAL = 12.0   # seconds with no game "state" event before an idle animation plays

