import cv2
import numpy as np

def evaluate_matches(
    keypoints1,
    keypoints2,
    matches,
    homography,
    reprojection_threshold=3.0
):
    """
    Evaluate matches using the ground-truth homography.

    A match is correct when the distance between the projected
    keypoint from image 1 and the matched keypoint in image 2
    is below the reprojection threshold.

    Returns a dictionary containing the metrics and per-match errors.
    """

    if homography is None or homography.shape != (3, 3):
        raise ValueError("Homography must be a 3x3 matrix.")

    if reprojection_threshold <= 0:
        raise ValueError("The reprojection threshold must be positive.")

    n_keypoints1 = len(keypoints1)
    n_keypoints2 = len(keypoints2)
    n_matches = len(matches)

    if n_matches == 0:
        return {
            "keypoints_image1": n_keypoints1,
            "keypoints_image2": n_keypoints2,
            "putative_matches": 0,
            "correct_matches": 0,
            "incorrect_matches": 0,
            "precision": 0.0,
            "pmr": 0.0,
            "matching_score": 0.0,
            "mean_reprojection_error": None,
            "mean_correct_match_error": None,
            "correct_match_rate": 0.0,
            "match_errors": []
        }

    # Get the matched keypoint coordinates in image 1.
    points1 = np.float32([
        keypoints1[match.queryIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    # Get the matched keypoint coordinates in image 2.
    points2 = np.float32([
        keypoints2[match.trainIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    # Project points from image 1 into image 2.
    projected_points = cv2.perspectiveTransform(
        points1,
        homography
    )

    projected_points = projected_points.reshape(-1, 2)
    points2 = points2.reshape(-1, 2)

    # Calculate the reprojection error for every match.
    errors = np.linalg.norm(
        projected_points - points2,
        axis=1
    )

    # A match is correct if its error is below the threshold.
    correct_mask = errors < reprojection_threshold

    n_correct = int(np.sum(correct_mask))
    n_incorrect = n_matches - n_correct

    precision = n_correct / n_matches

    # Initial reference: keypoints detected in image 1.
    pmr = n_matches / n_keypoints1 if n_keypoints1 else 0.0

    matching_score = (
        n_correct / n_keypoints1
        if n_keypoints1 else 0.0
    )

    mean_error = float(np.mean(errors))
    mean_correct_error = (
        float(np.mean(errors[correct_mask]))
        if n_correct > 0 else None
    )

    return {
        "keypoints_image1": n_keypoints1,
        "keypoints_image2": n_keypoints2,
        "putative_matches": n_matches,
        "correct_matches": n_correct,
        "incorrect_matches": n_incorrect,
        "precision": precision,
        "pmr": pmr,
        "matching_score": matching_score,
        "mean_reprojection_error": mean_error,
        "mean_correct_match_error": mean_correct_error,
        "correct_match_rate": precision,
        "match_errors": [
            {
                "queryIdx": int(match.queryIdx),
                "trainIdx": int(match.trainIdx),
                "error_pixels": float(error),
                "correct": bool(is_correct)
            }
            for match, error, is_correct in zip(
                matches, errors, correct_mask
            )
        ]
    }
