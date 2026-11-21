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