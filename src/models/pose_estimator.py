import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from dataclasses import dataclass

@dataclass
class BodyRatios:
    swr: float
    whr: float
    shoulder_width: float
    hip_width: float
    height: float

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)

    def extract_ratios(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        # Get key points
        ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = lm[self.mp_pose.PoseLandmark.NOSE]
        ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE]

        # Calculate ratios
        shoulder_w = np.sqrt((rs.x - ls.x)**2 + (rs.y - ls.y)**2)
        hip_w = np.sqrt((rh.x - lh.x)**2 + (rh.y - lh.y)**2)
        height = np.sqrt((ankle.x - nose.x)**2 + (ankle.y - nose.y)**2)

        return BodyRatios(
            swr=shoulder_w/hip_w if hip_w>0 else 0,
            whr=hip_w/height if height>0 else 0,
            shoulder_width=shoulder_w,
            hip_width=hip_w,
            height=height
        )