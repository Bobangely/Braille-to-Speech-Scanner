"""Bounded optical-flow alignment; reject uncertain motion instead of drifting."""

import cv2
import numpy as np


def tracking_gray(image, max_width=640):
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def estimate_motion(reference, current):
    """Return reference-to-current affine motion in small-image coordinates."""
    if reference.shape != current.shape:
        return None
    if np.array_equal(reference, current):
        return np.eye(3, dtype=np.float64)
    points = cv2.goodFeaturesToTrack(reference, maxCorners=100, qualityLevel=.02,
                                   minDistance=8, blockSize=5)
    if points is None or len(points) < 3:
        return None
    sparse = len(points) < 8
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
    forward, status, error = cv2.calcOpticalFlowPyrLK(reference, current, points, None, **options)
    if forward is None or status is None or error is None or not np.isfinite(forward).all():
        return None
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, reference, forward, None, **options)
    if backward is None or reverse_status is None:
        return None
    valid = (status.ravel() == 1) & (reverse_status.ravel() == 1)
    valid &= np.linalg.norm(points - backward, axis=2).ravel() < 1.5
    valid &= error.ravel() < 25
    if sparse:
        # Short words cannot supply eight corners. Accept translation only when
        # every observed corner agrees; do not fit rotation/scale to scant data.
        if not np.all(valid):
            return None
        source, target = points[:, 0], forward[:, 0]
        if np.linalg.svd(source - source.mean(axis=0), compute_uv=False)[1] < 1:
            return None  # A single row provides insufficient spatial evidence.
        displacement = target - source
        shift = np.median(displacement, axis=0)
        if np.linalg.norm(shift) > 8 or np.max(np.linalg.norm(displacement - shift, axis=1)) > .75:
            return None
        affine = np.array([[1., 0., shift[0]], [0., 1., shift[1]]])
    else:
        if np.count_nonzero(valid) < 8 or np.mean(valid) < .6:
            return None
        source, target = points[valid, 0], forward[valid, 0]
        affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                    ransacReprojThreshold=2, maxIters=500)
        if affine is None or not np.isfinite(affine).all() or inliers is None or np.mean(inliers) < .75:
            return None
    # This tracker intentionally handles small handheld movements, not scene changes.
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    angle = abs(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
    if not .9 <= scale <= 1.1 or angle > 12:
        return None
    if np.linalg.norm(affine[:, 2]) > .2 * np.hypot(*reference.shape):
        return None
    # Check actual pixels too: repeated Braille patterns can fool feature matches.
    size = (current.shape[1], current.shape[0])
    warped = cv2.warpAffine(reference, affine, size)
    support = cv2.warpAffine(np.full_like(reference, 255), affine, size) > 250
    ink = support & ((warped < 200) | (current < 200))
    if np.count_nonzero(ink) < 32:
        return None
    # With few dots, one changed dot may mean a different character.
    if np.mean(np.abs(warped.astype(float)[ink] - current[ink])) > (20 if sparse else 30):
        return None
    return np.vstack((affine, [0., 0., 1.]))


def full_size_motion(matrix, image_shape, small_shape):
    scale = np.diag([small_shape[1] / image_shape[1], small_shape[0] / image_shape[0], 1.])
    return np.linalg.inv(scale) @ matrix @ scale
