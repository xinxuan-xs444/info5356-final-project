"""Module 3 of Quizzy Mini's Math Showdown: hand-raise buzz-in detection and seat tracking."""

from .buzzer import BuzzArbiter, BuzzConfig
from .hand_raise import HandRaiseRule
from .pose import PoseEstimator, YoloPoseEstimator
from .seats import SeatMap
from .service import VisionService
from .sources import FrameSource, ListSource, OpenCVSource, ReachyMiniSource, VideoFileSource, WebcamSource
from .datatypes import BuzzResult, Frame, FrameResult, HandRaise, PersonPose, SeatObservation

__all__ = [
    "BuzzArbiter", "BuzzConfig", "BuzzResult", "Frame", "FrameResult", "FrameSource", "HandRaise",
    "HandRaiseRule", "ListSource", "OpenCVSource", "PersonPose", "PoseEstimator", "ReachyMiniSource",
    "SeatMap", "SeatObservation", "VideoFileSource", "VisionService", "WebcamSource", "YoloPoseEstimator",
]
