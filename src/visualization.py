import cv2

def draw_keypoints(image, keypoints):
    """
    Draw detected keypoints on an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in BGR or grayscale format.
    keypoints : list
        Detected OpenCV keypoints.

    Returns
    -------
    output : numpy.ndarray
        Image with keypoints drawn.
    """
    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return output


def draw_matches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches,
    max_matches=50
):
    """
    Draw matches between two images.

    Parameters
    ----------
    image1, image2 : numpy.ndarray
        Input images in BGR or grayscale format.
    keypoints1, keypoints2 : list
        Keypoints detected in each image.
    matches : list
        Matches between the two images.
    max_matches : int
        Maximum number of matches to display.

    Returns
    -------
    output : numpy.ndarray
        Side-by-side image showing the matches.
    """
    matches_to_draw = sorted(
        matches,
        key=lambda match: match.distance
    )[:max_matches]

    output = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return output
