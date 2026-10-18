import cv2

def match_descriptors(
    descriptors1,
    descriptors2,
    ratio_threshold=0.8,
    norm_type="L2"
):
    """
    Match descriptors with Lowe's ratio test and one-to-one filtering.

    norm_type:
        L2      - SIFT, KAZE
        HAMMING - ORB, BRISK, FAST+BRISK
    """

    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) == 0 or len(descriptors2) < 2:
        return []

    if norm_type.upper() == "L2":
        norm = cv2.NORM_L2
    elif norm_type.upper() == "HAMMING":
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    matcher = cv2.BFMatcher(norm, crossCheck=False)

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    ratio_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        best, second = pair

        if best.distance < ratio_threshold * second.distance:
            ratio_matches.append(best)

    ratio_matches.sort(key=lambda match: match.distance)

    unique_matches = []
    used_train_indices = set()

    for match in ratio_matches:
        if match.trainIdx in used_train_indices:
            continue

        unique_matches.append(match)
        used_train_indices.add(match.trainIdx)

    return unique_matches
