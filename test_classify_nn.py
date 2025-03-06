import sys
import numpy as np
from sklearn.model_selection import train_test_split

from Gesture import GestureSet, Sequence, Frame
from classify_nn import classify_nn
from normalize_frames import normalize_frames
from load_gestures import load_gestures
import random


def test_classify_nn(num_frames, ratio, seed=142):
    """
    Tests classify_nn function.
    Splits gesture data into training and testing sets and computes the accuracy of classify_nn()
    :param num_frames: the number of frames to normalize to
    :param ratio: percentage to be used for training
    :return: the accuracy of classify_nn()
    """

    gesture_sets = load_gestures()
    norm_gesture_sets = normalize_frames(gesture_sets, num_frames)

    train_gesture_sets = []
    test_sequences = []
    test_labels = []

    # Split into training and testing sets
    for gesture_set in norm_gesture_sets:
        sequences = gesture_set.sequences
        random.shuffle(sequences)  # Shuffle sequences to avoid bias

        split_idx = int(len(sequences) * ratio)  # Compute split index
        train_sequences = sequences[:split_idx]
        test_sequences.extend(sequences[split_idx:])  # Keep test sequences separate
        test_labels.extend([gesture_set.label] * (len(sequences) - split_idx))

        # Store only the training data per label
        train_gesture_sets.append(GestureSet(train_sequences, gesture_set.label))

    # Classify each test sequence
    correct = 0
    for test_seq, actual_label in zip(test_sequences, test_labels):
        predicted_label = classify_nn(test_seq, train_gesture_sets)
        if predicted_label == actual_label:
            correct += 1

    # Compute accuracy
    accuracy = correct / len(test_sequences) if test_sequences else 0
    return accuracy


# if len(sys.argv) != 3:
#     raise ValueError(
#         "Error! Give normalized frame number and test/training ratio after filename in command. \n"
#         "e.g. python test_nn.py 20 0.4"
#     )

# num_frames = int(sys.argv[1])
# ratio = float(sys.argv[2])

# accuracy = test_classify_nn(num_frames, ratio)
# print("Accuracy: ", accuracy)
