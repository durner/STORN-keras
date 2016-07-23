import numpy as np
from qlearning4k.games.game import Game
from greenarm.util import pad_sequences_3d
from greenarm.util import get_logger

logger = get_logger(__name__)

actions = {0: 'detect', 1: 'idle'}

CORRECT_DETECTION_SCORE = 15
INCORRECT_DETECTION_SCORE = -2
CORRECT_IDLE_SCORE = 0
INCORRECT_IDLE_SCORE = -10

ALL_CORRECT_SCORE = 30
ALL_CORRECT_NO_ANOMALIES_SCORE = 20


class AnomalyDetection(Game):
    def __init__(self, sequences, anomalies, valid_window=(-5, 10), sequence_like_states=True):
        self.sequence_like_states = sequence_like_states
        self.valid_window = valid_window
        self.sequences = sequences
        self.sequence_length = sequences.shape[1]
        self.current_sequence = None
        self.current_sequence_idx = -1
        self.anomalies = anomalies

        self.current_timestamp = 0
        self.num_spotted_correctly = 0
        self.score = 0

        self.anomalies_to_detect = None
        self.anomalies_to_detect_buffer = None

        self.detected_anomalies = []
        self.reset()

    @property
    def name(self):
        return "AnomalyDetection"

    @property
    def nb_actions(self):
        return 2

    def play(self, action):
        assert action in range(2), "Invalid action."

        if self.anomalies_to_detect_buffer:
            self.clear_old_anomalies()

        if action == 0:
            self.execute_idle()
        else:
            self.execute_detection()

        self.current_timestamp += 1

        if self.is_over():
            if self.is_won():
                if self.num_spotted_correctly == 0:
                    self.score += ALL_CORRECT_NO_ANOMALIES_SCORE
                    logger.debug("All correct, no anomalies!")
                else:
                    self.score += ALL_CORRECT_SCORE
                    logger.debug("All correct!")

    def clear_old_anomalies(self):
        self.anomalies_to_detect_buffer = [
            anomaly for anomaly in self.anomalies_to_detect_buffer
            if self.current_timestamp < (anomaly + self.valid_window[1])
            # if anomaly > (self.current_timestamp + self.valid_window[1])
        ]

    def current_timestamp_is_anomalous(self):
        if not self.anomalies_to_detect_buffer:
            return False

        start = int(self.anomalies_to_detect_buffer[0] + self.valid_window[0])
        end = int(self.anomalies_to_detect_buffer[0] + self.valid_window[1])
        return start <= self.current_timestamp < end

    def current_timestamp_is_last_anomalous(self):
        if not self.anomalies_to_detect_buffer:
            return False

        end = int(self.anomalies_to_detect_buffer[0]) + self.valid_window[1]
        return self.current_timestamp == end

    def execute_idle(self):
        if self.current_timestamp_is_last_anomalous():
            self.score += INCORRECT_IDLE_SCORE
            logger.debug("Missed an anomaly! (%s)" % self.anomalies_to_detect_buffer[0])
        else:
            self.score += CORRECT_IDLE_SCORE

    def execute_detection(self):
        # track the detection in any case
        self.detected_anomalies[self.current_sequence_idx].append(self.current_timestamp)
        if self.current_timestamp_is_anomalous():
            logger.debug("Correctly spotted an anomaly (%s)!" % self.anomalies_to_detect_buffer[0])
            self.num_spotted_correctly += 1
            self.score += CORRECT_DETECTION_SCORE
            # remove anomaly from buffer once we've detected it
            self.anomalies_to_detect_buffer.pop(0)
        else:
            self.score += INCORRECT_DETECTION_SCORE

    def get_state(self):
        # TODO we could also return the number of already detected anomalies, or interlace them with the sequence
        if self.sequence_like_states:
            # return sequence up to the current timestamp
            return pad_sequences_3d([self.current_sequence[:self.current_timestamp]], self.sequence_length)
        else:
            return self.current_sequence[self.current_timestamp]

    def get_score(self):
        return self.score

    def reset(self):
        self.num_spotted_correctly = 0
        self.score = 0

        self.current_sequence_idx += 1
        self.current_sequence_idx %= self.sequences.shape[0]  # rotate sequences

        if self.current_sequence_idx % 50 == 0:
            logger.debug(
                "Getting sequence %s / %s. Current score: %s." % (
                    self.current_sequence_idx, self.sequences.shape[0], self.score
                )
            )

        self.current_timestamp = 0
        self.current_sequence = self.sequences[self.current_sequence_idx]
        self.anomalies_to_detect = self.anomalies[self.current_sequence_idx]
        self.anomalies_to_detect_buffer = self.anomalies[self.current_sequence_idx][:]
        self.detected_anomalies.append([])

    def is_current_sequence_over(self):
        return self.current_timestamp == self.sequence_length

    def is_over(self):
        if self.current_timestamp == self.sequence_length - 1:
            logger.debug("Game is over, scored %s points" % self.score)
            return True
        else:
            return False

    def is_won(self):
        num_spotted_incorrectly = len(self.detected_anomalies) - self.num_spotted_correctly
        return self.num_spotted_correctly == len(self.anomalies_to_detect) and num_spotted_incorrectly == 0
