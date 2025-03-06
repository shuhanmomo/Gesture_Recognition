import numpy as np


class GestureSet:
    """
    A set of the same gesture, repeated by different users
    """

    def __init__(self, sequences, label):
        """
        :param sequences: List of Sequences
        :param label: Number associated with gesture
        """
        self.sequences = sequences
        self.label = label


class Sequence:
    """
    A sequence is a single gesture composed of ordered frames
    """

    def __init__(self, frames, label):
        self.frames = frames
        self.label = label


class Frame:
    """
    A data structure to hold a frame of a gesture - (x,y,z) points
    """

    def __init__(self, frame, prev_frame=None, delta_t=1.0):
        """
        :param frame: Current frame (list of joint (x,y,z) positions).
        :param prev_frame: Previous frame for computing velocity-based features.
        :param delta_t: Time difference between frames (assumed constant for simplicity).
        """
        self.frame = np.array(frame)
        self.prev_frame = np.array(prev_frame) if prev_frame is not None else None
        self.delta_t = delta_t  # Time step for velocity calculations

    def head(self):
        return self.frame[0:3]

    def neck(self):
        return self.frame[3:6]

    def left_shoulder(self):
        return self.frame[6:9]

    def left_elbow(self):
        return self.frame[9:12]

    def left_hand(self):
        return self.frame[12:15]

    def right_shoulder(self):
        return self.frame[15:18]

    def right_elbow(self):
        return self.frame[18:21]

    def right_hand(self):
        return self.frame[21:24]

    def torso(self):
        return self.frame[24:27]

    def left_hip(self):
        return self.frame[27:30]

    def right_hip(self):
        return self.frame[30:]

    def compute_joint_velocity(self):
        """
        Computes velocity for each joint: v = (current_position - previous_position) / delta_t
        """
        if self.prev_frame is None:
            return np.zeros_like(self.frame)  # No previous frame, return zero velocity

        velocity = (self.frame - self.prev_frame) / self.delta_t
        return velocity.flatten()

    def compute_joint_angles(self):
        """
        Computes angles between adjacent joints using cosine similarity.
        Returns a list of computed angles in degrees.
        """

        def angle_between(v1, v2):
            """Computes the angle between two vectors in degrees."""
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norms == 0:  # Prevent division by zero
                return 0.0
            cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        # Define key joint pairs for angle computation
        joint_pairs = [
            (self.neck(), self.left_shoulder()),
            (self.neck(), self.right_shoulder()),
            (self.left_shoulder(), self.left_elbow()),
            (self.right_shoulder(), self.right_elbow()),
            (self.left_elbow(), self.left_hand()),
            (self.right_elbow(), self.right_hand()),
            (self.torso(), self.left_hip()),
            (self.torso(), self.right_hip()),
            (self.left_hip(), self.right_hip()),
        ]

        angles = [angle_between(j1 - j2, j2) for j1, j2 in joint_pairs]
        return np.array(angles)

    def compute_angular_velocity(self):
        """
        Computes the rate of change of angles between frames.
        Angular velocity = (current_angle - previous_angle) / delta_t
        """
        if self.prev_frame is None:
            return np.zeros(
                len(self.compute_joint_angles())
            )  # No previous frame, return zero angular velocity

        prev_angles = Frame(self.prev_frame).compute_joint_angles()
        current_angles = self.compute_joint_angles()

        angular_velocity = (current_angles - prev_angles) / self.delta_t
        return angular_velocity

    def augment_features(self):
        """
        Appends computed features (velocity, joint angles, angular velocity) to self.frame.
        """
        velocity_features = self.compute_joint_velocity()
        angle_features = self.compute_joint_angles()
        angular_velocity_features = self.compute_angular_velocity()

        # Append new features to the frame
        self.frame = np.concatenate(
            [self.frame, velocity_features, angle_features, angular_velocity_features]
        )
        return self.frame


# Main Script to load Gestures.MAT into python objects
# Gestures = scipy.io.loadmat('gesture_dataset.mat')
# gestures = Gestures['gestures']
# gesture_sets = [GestureSet(g) for g in gestures[0]]
