import cv2

METHODS = {
    "SIFT": "L2",
    "ORB": "HAMMING",
    "KAZE": "L2",
    "BRISK": "HAMMING",
    "FAST_BRISK": "HAMMING",
}


def detect_and_describe(image, method="SIFT"):
    """
    Detect keypoints and compute descriptors.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input image.
    method : str
        SIFT, ORB, KAZE, BRISK or FAST_BRISK.

    Returns
    -------
    keypoints : list
        Detected keypoints.
    descriptors : numpy.ndarray or None
        Feature descriptors.
    """

    if image is None or image.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    method = method.upper()

    if method == "SIFT":
        detector = cv2.SIFT_create()
        return detector.detectAndCompute(image, None)

    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
        return detector.detectAndCompute(image, None)

    elif method == "KAZE":
        detector = cv2.KAZE_create()
        return detector.detectAndCompute(image, None)

    elif method == "BRISK":
        detector = cv2.BRISK_create()
        return detector.detectAndCompute(image, None)

    elif method == "FAST_BRISK":
        fast = cv2.FastFeatureDetector_create(
            threshold=20,
            nonmaxSuppression=True
        )

        brisk = cv2.BRISK_create()

        keypoints = fast.detect(image, None)

        if not keypoints:
            return [], None

        keypoints, descriptors = brisk.compute(
            image,
            keypoints
        )

        return keypoints, descriptors

    else:
        raise ValueError(f"Unsupported method: {method}")


def detect_sift(image, nfeatures=0):
    """
    Keep compatibility with the initial SIFT implementation.
    """
    if image is None or image.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    return sift.detectAndCompute(image, None)
