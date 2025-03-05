from Gesture import GestureSet, Sequence, Frame
import numpy as np


def normalize_frames(gesture_sets, num_frames):
    """
    Normalizes the number of Frames in each Sequence in each GestureSet
    :param gesture_sets: the list of GesturesSets
    :param num_frames: the number of frames to normalize to
    :return: a list of GestureSets where all Sequences have the same number of Frames
    """

    for gset in gesture_sets:
        for seq in gset.sequences:
            N_old = len(seq.frames)
            N_new = num_frames
            if N_old == N_new:
                continue
            indices_old = np.linspace(0, N_old - 1, num=N_old)
            indices_new = np.linspace(0, N_old - 1, num=N_new)
            seq_data = np.array([frame.frame for frame in seq.frames])
            D = seq_data.shape[1]

            resampled_data = np.array(
                [
                    np.interp(indices_new, indices_old, seq_data[:, dim])
                    for dim in range(D)
                ]
            ).T
            seq.frames = [Frame(resampled_data[i].tolist()) for i in range(N_new)]

    return gesture_sets
