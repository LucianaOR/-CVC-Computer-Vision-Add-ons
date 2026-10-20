
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from src.experiments import load_image, load_homography
from src.evaluation import evaluate_matches
from src.visualization import draw_keypoints, draw_matches


class SuperPointLightGlue:
    """
    SuperPoint feature extraction + LightGlue matching.

    The models are initialized once and reused across image pairs.
    """

    def __init__(self, device="cpu", max_keypoints=2048):
        self.device = torch.device(device)

        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features="superpoint"
        ).eval().to(self.device)

    def prepare_image(self, image_color):
        """Convert an OpenCV BGR image to a normalized RGB tensor."""

        image_rgb = cv2.cvtColor(
            image_color,
            cv2.COLOR_BGR2RGB
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb)
        )

        tensor = tensor.permute(2, 0, 1).float() / 255.0

        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def extract(self, image_tensor):
        """Extract SuperPoint keypoints and descriptors."""

        start = perf_counter()
        features = self.extractor.extract(image_tensor)
        elapsed_ms = (perf_counter() - start) * 1000

        return features, elapsed_ms

    @torch.inference_mode()
    def match(self, features1, features2):
        """Match two sets of SuperPoint features using LightGlue."""

        start = perf_counter()

        output = self.matcher({
            "image0": features1,
            "image1": features2
        })

        elapsed_ms = (perf_counter() - start) * 1000

        features1, features2, output = [
            rbd(value)
            for value in (features1, features2, output)
        ]

        return features1, features2, output, elapsed_ms


def tensor_keypoints_to_cv2(keypoints):
    """Convert an Nx2 tensor into OpenCV KeyPoint objects."""

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()

    return [
        cv2.KeyPoint(
            float(point[0]),
            float(point[1]),
            1.0
        )
        for point in keypoints
    ]


def lightglue_matches_to_cv2(matches):
    """
    Convert LightGlue's Nx2 index pairs into cv2.DMatch objects.
    """

    if isinstance(matches, torch.Tensor):
        matches = matches.detach().cpu().numpy()

    converted = []

    for index1, index2 in matches:
        converted.append(
            cv2.DMatch(
                int(index1),
                int(index2),
                0.0
            )
        )

    return converted


def tensor_memory_bytes(tensor):
    """Return the memory occupied by a tensor."""

    if tensor is None:
        return 0

    return tensor.element_size() * tensor.nelement()


def run_superpoint_lightglue_pair(
    sequence_dir,
    target_index,
    model,
    figures_dir=None,
    reprojection_threshold=3.0
):
    """
    Evaluate SuperPoint + LightGlue on one HPatches image pair.
    """

    sequence_dir = Path(sequence_dir)

    image1_color, _ = load_image(
        sequence_dir / "1.ppm"
    )

    image2_color, _ = load_image(
        sequence_dir / f"{target_index}.ppm"
    )

    homography = load_homography(
        sequence_dir,
        target_index
    )

    tensor1 = model.prepare_image(image1_color)
    tensor2 = model.prepare_image(image2_color)

    features1, extraction_time1_ms = model.extract(tensor1)
    features2, extraction_time2_ms = model.extract(tensor2)

    features1, features2, output, matching_time_ms = model.match(
        features1,
        features2
    )

    keypoints1 = tensor_keypoints_to_cv2(
        features1["keypoints"]
    )

    keypoints2 = tensor_keypoints_to_cv2(
        features2["keypoints"]
    )

    matches = lightglue_matches_to_cv2(
        output["matches"]
    )

    metrics = evaluate_matches(
        keypoints1,
        keypoints2,
        matches,
        homography,
        reprojection_threshold=reprojection_threshold
    )

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        prefix = (
            f"superpoint_lightglue_"
            f"{sequence_dir.name}_1_{target_index}"
        )

        keypoints_image = draw_keypoints(
            image1_color,
            keypoints1
        )

        matches_image = draw_matches(
            image1_color,
            keypoints1,
            image2_color,
            keypoints2,
            matches,
            max_matches=50
        )

        if not cv2.imwrite(
            str(figures_dir / f"{prefix}_keypoints.png"),
            keypoints_image
        ):
            raise IOError("Could not save keypoint visualization.")

        if not cv2.imwrite(
            str(figures_dir / f"{prefix}_matches.png"),
            matches_image
        ):
            raise IOError("Could not save match visualization.")

    descriptors1 = features1.get("descriptors")
    descriptors2 = features2.get("descriptors")

    descriptor_memory = (
        tensor_memory_bytes(descriptors1)
        + tensor_memory_bytes(descriptors2)
    )

    descriptor_dimension = (
        int(descriptors1.shape[-1])
        if descriptors1 is not None
        else 0
    )

    return {
        "sequence": sequence_dir.name,
        "transformation": (
            "illumination"
            if sequence_dir.name.startswith("i_")
            else "viewpoint"
        ),
        "image1": 1,
        "image2": target_index,
        "method": "SuperPoint+LightGlue",
        "keypoints_image1": metrics["keypoints_image1"],
        "keypoints_image2": metrics["keypoints_image2"],
        "putative_matches": metrics["putative_matches"],
        "correct_matches": metrics["correct_matches"],
        "incorrect_matches": metrics["incorrect_matches"],
        "precision": metrics["precision"],
        "pmr": metrics["pmr"],
        "matching_score": metrics["matching_score"],
        "mean_reprojection_error_px":
            metrics["mean_reprojection_error"],
        "mean_correct_match_error_px":
            metrics["mean_correct_match_error"],
        "detection_description_time1_ms":
            extraction_time1_ms,
        "detection_description_time2_ms":
            extraction_time2_ms,
        "matching_time_ms": matching_time_ms,
        "descriptor_dimension": descriptor_dimension,
        "descriptor_memory_bytes": descriptor_memory
    }
