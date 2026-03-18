import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import glob
import einops
import copy
from typing import Tuple

from omegaconf import open_dict
from hydra import compose, initialize
from feature_extractor.cutie.cutie.model.cutie import CUTIE
from feature_extractor.cutie.cutie.inference.inference_core import InferenceCore
from feature_extractor.cutie.cutie.inference.utils.args_utils import get_dataset_cfg
from feature_extractor.cutie.cuite_gui.interactive_utils import (
    image_to_torch,
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
    overlay_davis,
)


def generate_color_map(num_classes):
    """
    Generate a color map for visualization
    """
    hsv_colors = [(i / num_classes, 1.0, 1.0) for i in range(num_classes)]
    rgb_colors = [np.array([0, 0, 0], dtype=np.uint8)]  # Background color
    for h, s, v in hsv_colors:
        hsv_color = np.uint8([[[h * 180, s * 255, v * 255]]])  # OpenCV uses H: 0-180, S: 0-255, V: 0-255
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        rgb_colors.append(rgb_color)
    return np.array(rgb_colors, dtype=np.uint8)


def load_cuite(model_size):
    """
    Load the CUTIE model

    Parameters:
    model_size (str): The size of the model to load. Options are "small" and "base"
    """
    assert model_size in ["small", "base"], "model_size must be either 'small' or 'base'"
    with torch.inference_mode():
        initialize(version_base="1.3.2", config_path="cutie/config", job_name=f"eval_config_{model_size}")
        cfg = compose(config_name=f"eval_config_{model_size}")

        with open_dict(cfg):
            cfg["weights"] = f"feature_extractor/cutie/weights/cutie-{model_size}-mega.pth"

        data_cfg = get_dataset_cfg(cfg)

        # Load the network weights
        cutie = CUTIE(cfg).cuda().eval()
        model_weights = torch.load(cfg.weights)
        cutie.load_weights(model_weights)
    return cutie, cfg


def unique_mask_values(mask):
    # create a mapping from unique values to integers
    unique_values = np.unique(mask)
    value_to_int = {value: i for i, value in enumerate(unique_values)}

    # convert the mask to integers
    mask_converted = np.vectorize(value_to_int.get)(mask)
    return mask_converted


class CutieFeatureExtractor:  # object vectors
    def __init__(
        self, label_folder, num_objects, model_size, expected_resolution, resolution_scale_factor, frame_scale_method
    ):
        """
        Initialize the CutieFeatureExtractor
        Parameters:
            label_folder (str): The path to the folder containing the labeled images and masks
            num_objects (int): The number of objects in the env
            scale_method (str): The method to use for scaling the images. Options are "nearest" and "bilinear", this only applies to the frame rather than the mask
        """
        # check if the label folder exists
        assert os.path.exists(label_folder), f"{label_folder} does not exist"
        self.num_objects = num_objects
        self.resolution_scale_factor = resolution_scale_factor
        self.width, self.height = expected_resolution

        self.color_map = generate_color_map(num_objects)  # for visualization

        # Load the Cutie model
        print("loading cutie...")
        cutie, cutie_cfg = load_cuite(model_size)
        self.cuite_processor = InferenceCore(cutie, cfg=cutie_cfg)
        print("cutie loaded")

        # Load the reference frame and mask
        print("loading reference frame...")
        # get the number of labeled images
        num_images = len(glob.glob(f"{label_folder}/imgs/*.png"))

        if frame_scale_method == "nearest":
            self.frame_interpolation_method = cv2.INTER_NEAREST
        elif frame_scale_method == "bilinear":
            self.frame_interpolation_method = cv2.INTER_LINEAR
        else:
            raise ValueError(f"Unknown scale method: {frame_scale_method}")

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                for i in range(num_images):
                    ref_frame = cv2.imread(f"{label_folder}/imgs/{i}.png")
                    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)

                    ref_frame = cv2.resize(
                        ref_frame,
                        (
                            int(self.width * self.resolution_scale_factor),
                            int(self.height * self.resolution_scale_factor),
                        ),
                        interpolation=self.frame_interpolation_method,
                    )
                    ref_frame_torch = image_to_torch(ref_frame, device="cuda")

                    mask = np.array(Image.open(f"{label_folder}/masks/{i}.png"))
                    # Mask can't use bilinear interpolation
                    mask = cv2.resize(
                        mask,
                        (
                            int(self.width * self.resolution_scale_factor),
                            int(self.height * self.resolution_scale_factor),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    # mask = unique_mask_values(mask)
                    # now use absolute values for the mask
                    mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).cuda()
                    prediction = self.cuite_processor.step(
                        ref_frame_torch, mask_torch[1:], idx_mask=False, force_permanent=True
                    )

        print(f"reference frame loaded, {num_images} images/masks")

    def reset(self):
        # clear cutie memory per episode
        self.cuite_processor.clear_non_permanent_memory()

    def extract_features(self, frame) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract features from the frame
        Parameters:
            frame (np.ndarray): The frame to extract features from
        Returns:
            torch.Tensor: The object features, torch.Tensor of shape (num_objects, feature_size), on cuda
        """
        # cuite inference
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                frame = cv2.resize(
                    frame,
                    (int(self.width * self.resolution_scale_factor), int(self.height * self.resolution_scale_factor)),
                    interpolation=self.frame_interpolation_method,
                )
                frame_torch = image_to_torch(frame, device="cuda")
                prediction = self.cuite_processor.step(frame_torch)
                mask = torch_prob_to_numpy_mask(prediction)

                object_features = copy.deepcopy(
                    self.cuite_processor.network.object_transformer.query_post_process_cache
                )  # [Obj, 16, 256]

                # foreground feature, 8 is the first half of N=16, see eq(3) of Cutie paper
                object_features = einops.rearrange(
                    object_features[:, :8], "Obj halfN C -> Obj (halfN C)"
                )  # [Obj, 2048]

                # if defocus, set the object features to 0
                defocus_flag = copy.deepcopy(self.cuite_processor.network.object_transformer.defocus_cache)  # [Obj]
                object_features = object_features * einops.rearrange(defocus_flag, "Obj -> Obj 1")

        visualization_obs = np.concatenate(
            [frame, self.color_map[mask]], axis=0 if frame.shape[0] < frame.shape[1] else 1
        )
        return object_features, visualization_obs


class VisualMaskExtractor(CutieFeatureExtractor):  # visual + mask
    def __init__(self, state_resolution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_resolution = state_resolution

    def extract_features(self, frame) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract features from the frame
        Parameters:
            frame (np.ndarray): The frame to extract features from
        Returns:
            torch.Tensor: The object features, torch.Tensor of shape (num_objects, feature_size), on cuda
        """
        # cuite inference
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                frame = cv2.resize(
                    frame,
                    (int(self.width * self.resolution_scale_factor), int(self.height * self.resolution_scale_factor)),
                    interpolation=self.frame_interpolation_method,
                )
                frame_torch = image_to_torch(frame, device="cuda")
                prediction = self.cuite_processor.step(frame_torch)
                mask = torch_prob_to_numpy_mask(prediction)

                state_frame = F.interpolate(
                    frame_torch.unsqueeze(0), size=self.state_resolution, mode="bilinear", align_corners=False
                ).squeeze(0)
                state_mask = F.interpolate(
                    prediction.unsqueeze(0), size=self.state_resolution, mode="bilinear", align_corners=False
                ).squeeze(0)
                state_mask = F.one_hot(torch.argmax(state_mask, dim=0), num_classes=self.num_objects + 1)
                state_mask = einops.rearrange(state_mask, "H W C-> C H W")
                state_mask = state_mask[1:]  # remove background
                state = torch.cat([state_frame, state_mask], dim=0)

        visualization_obs = np.concatenate(
            [frame, self.color_map[mask]], axis=0 if frame.shape[0] < frame.shape[1] else 1
        )
        return state, visualization_obs


class VectorPlusVisualExtractor(CutieFeatureExtractor):  # visual + mask
    def __init__(self, state_resolution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_resolution = state_resolution

    def extract_features(self, frame) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract features from the frame
        Parameters:
            frame (np.ndarray): The frame to extract features from
        Returns:
            torch.Tensor: The object features, torch.Tensor of shape (num_objects, feature_size), on cuda
        """
        # cuite inference
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                frame = cv2.resize(
                    frame,
                    (int(self.width * self.resolution_scale_factor), int(self.height * self.resolution_scale_factor)),
                    interpolation=self.frame_interpolation_method,
                )
                frame_torch = image_to_torch(frame, device="cuda")
                prediction = self.cuite_processor.step(frame_torch)
                mask = torch_prob_to_numpy_mask(prediction)

                # object vector
                object_features = copy.deepcopy(
                    self.cuite_processor.network.object_transformer.query_post_process_cache
                )  # [Obj, 16, 256]
                # foreground feature, 8 is the first half of N=16, see eq(3) of Cutie paper
                object_features = einops.rearrange(
                    object_features[:, :8], "Obj halfN C -> Obj (halfN C)"
                )  # [Obj, 2048]
                # if defocus, set the object features to 0
                defocus_flag = copy.deepcopy(self.cuite_processor.network.object_transformer.defocus_cache)  # [Obj]
                object_features = object_features * einops.rearrange(defocus_flag, "Obj -> Obj 1")

                # visual observation
                state_frame = F.interpolate(
                    frame_torch.unsqueeze(0), size=self.state_resolution, mode="bilinear", align_corners=False
                ).squeeze(0)

        visualization_obs = np.concatenate(
            [frame, self.color_map[mask]], axis=0 if frame.shape[0] < frame.shape[1] else 1
        )
        return (object_features, state_frame), visualization_obs
