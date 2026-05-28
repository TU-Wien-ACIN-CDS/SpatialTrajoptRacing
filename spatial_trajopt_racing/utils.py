from __future__ import annotations
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion
from skimage.segmentation import flood_fill
from typing import Tuple
from dataclasses import dataclass
import yaml
import os
from PIL import Image
import numpy as np
import scipy
from skimage.morphology import skeletonize
import sys
import cv2


def gen_centerline_from_img(map_img: np.ndarray, thresold: float):
    # grayscale -> binary. Converts grey to black
    img_copy: Image = map_img.copy()
    img_copy[map_img <= 210.] = 0
    img_copy[map_img > 210.] = 1

    # Calculate Euclidean Distance Transform (tells us distance to nearest wall)
    dist_transform1 = cv2.distanceTransform(img_copy, distanceType=cv2.DIST_L2, maskSize=5)
    dist_transform2 = cv2.distanceTransform(1 - img_copy, distanceType=cv2.DIST_L2, maskSize=5)
    dist_transform = dist_transform1 - dist_transform2

    # Threshold the distance transform to create a binary image
    centers: np.ndarray = dist_transform > thresold * dist_transform.max()
    centerline: np.ndarray = skeletonize(centers)

    # Only put distance values directly on the centerline
    NON_EDGE = 0.0
    centerline_dist: np.ndarray = np.where(
        centerline, dist_transform, NON_EDGE)

    # Find a proper starting position for DFS
    starting_point: (int, int) | None = None
    for x in range(centerline_dist.shape[1]):
        for y in range(centerline_dist.shape[0]):
            if (centerline_dist[y][x] != NON_EDGE):
                starting_point = (x, y)
                break

        if starting_point is not None:
            break

    if starting_point is None:
        raise ValueError("Could not find a starting point for the DFS")

    # Use DFS to extract the outer edge
    sys.setrecursionlimit(20000)
    DIRECTIONS: list[(int, int)] = [(0, -1), (-1, 0),  (0, 1), (1, 0),
                                    (-1, 1), (-1, -1), (1, 1), (1, -1)]

    visited: dict[(int, int), bool] = {}
    centerline_points: list[(int, int)] = []
    track_widths: list[(int, int)] = []

    def dfs(point: (int, int)):
        if point in visited:
            return

        visited[point] = True

        x, y = point
        centerline_points.append(np.array(point))

        track_width = centerline_dist[y][x]
        track_widths.append((track_width, track_width))

        for dx, dy in DIRECTIONS:
            candidate_point = x + dx, y + dy
            candidate_x, candidate_y = candidate_point
            if (candidate_x < 0 or candidate_x >= centerline_dist.shape[1] or
                    candidate_y < 0 or candidate_y >= centerline_dist.shape[0]):
                continue

            candidate_dist = centerline_dist[candidate_y][candidate_x]
            if (candidate_dist != NON_EDGE and candidate_point not in visited):
                dfs(candidate_point)

    dfs(starting_point)

    centerline_points = np.array(centerline_points)
    track_widths = np.array(track_widths)

    return centerline_points