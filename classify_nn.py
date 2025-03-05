import operator
import math
import numpy as np


def get_sequence_array(sequence):
    return np.array([frame.frame for frame in sequence.frames])


def classify_nn(test_sequence, training_gesture_sets):
    """
    Classify test_sequence using nearest neighbors
    :param test_gesture: Sequence to classify
    :param training_gesture_sets: training set of labeled gestures
    :return: a classification label (an integer between 0 and 8)
    """
    test_seq = get_sequence_array(test_sequence)  # shape (N_seq,33)

    min_dist = float("inf")
    best_label = None
    for label, gesture_set in enumerate(training_gesture_sets):

        ges_arr = np.array(
            [get_sequence_array(seq) for seq in gesture_set.sequences]
        )  # (N_gset,N_seq,33)
        distances = np.linalg.norm(ges_arr - test_seq, axis=(1, 2))
        avg_dist = np.mean(distances)
        if avg_dist < min_dist:
            min_dist = avg_dist
            best_label = label
    return best_label
