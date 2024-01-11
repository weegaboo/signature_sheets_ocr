import cv2

import numpy as np

from typing import List
from collections import defaultdict
from .hough_lines import HoughLines


class Segmenter(HoughLines):
    """
        A subclass of HoughLines, designed for segmenting Hough lines in image processing.
        It segments lines based on their orientation using K-means clustering.

        Attributes:
        -----------
        cv_max_val : int
            The maximum value to use in certain OpenCV operations, typically set to 255 for binary images.

        Methods:
        --------
        run(lines: List, k: int = 2, **kwargs) -> List:
            Segments Hough lines into groups based on their orientation using K-means clustering.

        plot(image_table: np.ndarray, hough_lines_segmented: List) -> np.ndarray:
            Draws the segmented Hough lines on a copy of the input image and returns this modified image.
    """
    def __init__(self, line_coef: int = 1000, cv_max_val: int = 255):
        """
            Initializes the Segmenter class with specific parameters for Hough line segmentation.

            Parameters:
            -----------
            line_coef : int
                A coefficient to determine the length of the line segments for visualization (inherited).
            cv_max_val : int
                The maximum value for binary image thresholding.
        """
        super().__init__(line_coef)
        self.cv_max_val = cv_max_val

    @staticmethod
    def run(lines, k=2, **kwargs) -> List:
        """
            Segments a list of Hough lines into groups based on their orientation using K-means clustering.

            Parameters:
            -----------
            lines : List
                A list of Hough lines to be segmented.
            k : int, optional
                The number of clusters (orientations) to form (default is 2).
            **kwargs
                Additional keyword arguments for the K-means clustering algorithm.

            Returns:
            --------
            List
                A list of lists, where each sublist contains Hough lines belonging to one segment.
        """
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)
        angles = np.array([line[0][1] for line in lines])
        pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                        for angle in angles], dtype=np.float32)
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)
        segmented = defaultdict(list)
        for i, line in zip(range(len(lines)), lines):
            segmented[labels[i]].append(line)
        return list(segmented.values())

    def plot(self, image_table: np.ndarray, hough_lines_segmented: List):
        """
            Draws the segmented Hough lines on a copy of the provided image.

            Parameters:
            -----------
            image_table : np.ndarray
                The original image on which Hough lines will be plotted.
            hough_lines_segmented : List
                A list of lists of segmented Hough lines.

            Returns:
            --------
            np.ndarray
                The image with drawn segmented Hough lines.
        """
        hough_lines_segmented_img = image_table.copy()
        for index, item in enumerate(hough_lines_segmented):
            color = (0, self.cv_max_val, 0)
            for line in item:
                for rho, theta in line:
                    point_1, point_2 = self._get_coordinates_of_hough_line(rho, theta)
                    if index == 0:
                        point_2.x += 2000
                    cv2.line(hough_lines_segmented_img, (point_1.x, point_1.y), (point_2.x, point_2.y), color, 2)
        return hough_lines_segmented_img

