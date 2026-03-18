import torch
import cv2
import numpy as np
from PIL import Image
import os
import glob
import einops
import copy
from typing import Tuple


class ResizeObservation:
    def __init__(self, state_resolution):
        self.state_resolution = state_resolution

    def reset(self):
        return

    def extract_features(self, frame) -> Tuple[torch.Tensor, np.ndarray]:
        state_frame = cv2.resize(frame, self.state_resolution, interpolation=cv2.INTER_LINEAR)
        state_frame = einops.rearrange(state_frame, "H W C -> C H W")
        state_frame = torch.from_numpy(state_frame).float().to("cuda", non_blocking=True) / 255
        return state_frame, frame
