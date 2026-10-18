from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.detectors import detect_and_describe, METHODS
from src.matching import match_descriptors
from src.evaluation import evaluate_matches
from src.visualization import draw_keypoints, draw_matches

def load_image(image_path):
    """Load an image in color and grayscale."""
    image_color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_color is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    image_gray = cv2.cvtColor(
        image_color, cv2.COLOR_BGR2GRAY
    )

    return image_color, image_gray


def load_homography(sequence_dir, target_index):
    """Load HPatches ground-truth homography H_1_target."""

    candidates = [
        sequence_dir / f"H_1_{target_index}",
        sequence_dir / f"H_1_{target_index}.txt"
    ]

    for path in candidates:
        if path.exists():
            H = np.loadtxt(path)

            if H.shape != (3, 3):
                raise ValueError(f"Invalid homography: {path}")

            if not np.all(np.isfinite(H)):
                raise ValueError(f"Non-finite homography: {path}")

            if abs(np.linalg.det(H)) < 1e-12:
                raise ValueError(f"Singular homography: {path}")

            return H

    raise FileNotFoundError(
        f"Missing H_1_{target_index} in {sequence_dir}"
    )


def run_feature_pair(
    sequence_dir,
    target_index,
    method="SIFT",
    figures_dir=None,
    ratio_threshold=0.8,
    reprojection_threshold=3.0
):
    """
    Detect, describe, match and evaluate one HPatches pair.
    """

    sequence_dir = Path(sequence_dir)
    method = method.upper()

    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    image1_color, image1_gray = load_image(
        sequence_dir / "1.ppm"
    )
    image2_color, image2_gray = load_image(
        sequence_dir / f"{target_index}.ppm"
    )

    H = load_homography(sequence_dir, target_index)

    start = perf_counter()
    keypoints1, descriptors1 = detect_and_describe(
        image1_gray, method
    )
    time1_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    keypoints2, descriptors2 = detect_and_describe(
        image2_gray, method
    )
    time2_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    matches = match_descriptors(
        descriptors1,
        descriptors2,
        ratio_threshold=ratio_threshold,
        norm_type=METHODS[method]
    )
    matching_time_ms = (perf_counter() - start) * 1000

    metrics = evaluate_matches(
        keypoints1,
        keypoints2,
        matches,
        H,
        reprojection_threshold=reprojection_threshold
    )

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        prefix = (
            f"{method.lower()}_"
            f"{sequence_dir.name}_1_{target_index}"
        )

        kp_image = draw_keypoints(
            image1_color, keypoints1
        )

        match_image = draw_matches(
            image1_color,
            keypoints1,
            image2_color,
            keypoints2,
            matches,
            max_matches=50
        )

        if not cv2.imwrite(
            str(figures_dir / f"{prefix}_keypoints.png"),
            kp_image
        ):
            raise IOError("Failed to save keypoint image.")

        if not cv2.imwrite(
            str(figures_dir / f"{prefix}_matches.png"),
            match_image
        ):
            raise IOError("Failed to save matches image.")

    descriptor_dimension = (
        int(descriptors1.shape[1])
        if descriptors1 is not None
        else 0
    )

    descriptor_memory_bytes = sum(
        descriptor.nbytes
        for descriptor in (descriptors1, descriptors2)
        if descriptor is not None
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
        "method": method,
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
        "detection_description_time1_ms": time1_ms,
        "detection_description_time2_ms": time2_ms,
        "matching_time_ms": matching_time_ms,
        "descriptor_dimension": descriptor_dimension,
        "descriptor_memory_bytes": descriptor_memory_bytes
    }
