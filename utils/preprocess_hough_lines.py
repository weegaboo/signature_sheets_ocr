import cv2
import math

import numpy as np

from typing import List
from .point import Point
from .hough_lines import HoughLines

class PreprocessHough(HoughLines):
    """
        A subclass of HoughLines, designed for preprocessing Hough lines in image processing.
        It includes functionality for calculating line intensities and filtering lines based on
        intensity thresholds.

        Attributes:
        -----------
        cv_thresh : int
            Threshold value for binary thresholding in image preprocessing.
        cv_max_val : int
            Maximum value to use in certain OpenCV operations, typically set to 255 for binary images.

        Methods:
        --------
        _get_line_points(image: np.ndarray, point_1: Point, point_2: Point) -> np.ndarray:
            Calculates pixel coordinates and intensities along a line between two points in an image.

        _get_parallel_line(point_1: Point, point_2: Point, offset: int) -> Tuple[Point, Point]:
            Computes a line parallel to the given line at a specified offset.

        _get_avg_intensities(image: np.ndarray, point_1: Point, point_2: Point) -> float:
            Calculates the average intensity of pixels along a line between two points.

        compute_hisogram_of_avg_pixel_values(values: np.ndarray) -> float:
            Computes the average value from an array, typically used for intensity analysis.

        filter_lines_by_intensity(grey: np.ndarray, hough_lines_filtered_by_orientation: List) -> np.ndarray:
            Filters Hough lines based on their intensity, returning a subset of the original lines.
    """
    def __init__(self,
                 line_coef: int = 1000,
                 cv_thresh: int = 128,
                 cv_max_val: int = 255):
        """
            Initializes the PreprocessHough class with specific parameters for Hough line preprocessing.

            Parameters:
            -----------
            line_coef : int
                A coefficient to determine the length of the line segments for visualization (inherited).
            cv_thresh : int
                The threshold value for binary thresholding in image preprocessing.
            cv_max_val : int
                The maximum value for binary image thresholding.
        """
        super().__init__(line_coef)
        self.cv_thresh = cv_thresh
        self.cv_max_val = cv_max_val

    @staticmethod
    def _get_line_points(image: np.ndarray, point_1: Point, point_2: Point):
        """
            Produces an array of coordinates and intensities for each pixel in a line between two points.

            Parameters:
            -----------
            image : np.ndarray
                The image being processed.
            point_1 : Point
                The first point (x, y) defining the line.
            point_2 : Point
                The second point (x, y) defining the line.

            Returns:
            --------
            np.ndarray
                An array of the coordinates and intensities of each pixel along the line.
        """
        height, width = image.shape[:2]
        dX = point_2.x - point_1.x
        dY = point_2.y - point_1.y
        dXa = np.abs(dX)
        dYa = np.abs(dY)
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)
        # Obtain coordinates along the line using a form of Bresenham's algorithm
        # negY = point_1.y > point_2.y
        # negX = point_1.x > point_2.x
        if point_1.x == point_2.x:
            itbuffer[:, 0] = point_1.x
            itbuffer[:, 1] = (
                np.arange(point_1.y - 1, point_1.y - dYa - 1, -1)
                if point_1.y > point_2.y
                else np.arange(point_1.y + 1, point_1.y + dYa + 1)
            )
        elif point_1.y == point_2.y:
            itbuffer[:, 1] = point_1.y
            if point_1.x > point_2.x:
                itbuffer[:, 0] = np.arange(point_1.x - 1, point_1.x - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(point_1.x + 1, point_1.x + dXa + 1)
        elif dYa > dXa:
            slope = dX / dY
            if point_1.y > point_2.y:
                itbuffer[:, 1] = np.arange(point_1.y - 1, point_1.y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(point_1.y + 1, point_1.y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - point_1.y)).astype(np.int_) + point_1.x
        else:
            slope = dY / dX
            if point_1.x > point_2.x:
                itbuffer[:, 0] = np.arange(point_1.x - 1, point_1.x - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(point_1.x + 1, point_1.x + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - point_1.x)).astype(np.int_) + point_1.y
        x_column = itbuffer[:, 0]
        y_column = itbuffer[:, 1]
        itbuffer = itbuffer[(x_column >= 0) & (y_column >= 0) & (x_column < width) & (y_column < height)]
        itbuffer[:, 2] = image[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
        return itbuffer

    @staticmethod
    def _get_parallel_line(point_1: Point, point_2: Point, offset):
        """
            Computes a line parallel to a given line, offset by a specified distance.

            Parameters:
            -----------
            point_1 : Point
                The first point (x, y) of the original line.
            point_2 : Point
                The second point (x, y) of the original line.
            offset : int
                The distance by which the parallel line is offset from the original line.

            Returns:
            --------
            Tuple[Point, Point]
                Two points defining the parallel line.
        """
        l = math.sqrt(
            (point_1.x - point_2.x) * (point_1.x - point_2.x) + (point_1.y - point_2.y) * (point_1.y - point_2.y)
        )
        x1_p = int(point_1.x + offset * (point_2.y - point_1.y) / l)
        y1_p = int(point_1.y + offset * (point_1.x - point_2.x) / l)
        parallel_point_1 = Point(x=x1_p, y=y1_p)
        x2_p = int(point_2.x + offset * (point_2.y - point_1.y) / l)
        y2_p = int(point_2.y + offset * (point_1.x - point_2.x) / l)
        parallel_point_2 = Point(x=x2_p, y=y2_p)
        return parallel_point_1, parallel_point_2

    def _get_avg_intensities(self, image, point_1, point_2):
        """
            Calculates the average intensity of pixels along a line between two points in an image.

            Parameters:
            -----------
            image : np.ndarray
                The image from which intensities are calculated.
            point_1 : Point
                The first point (x, y) defining the line.
            point_2 : Point
                The second point (x, y) defining the line.

            Returns:
            --------
            float
                The average intensity of the pixels along the line.
        """
        line_iterator = self._get_line_points(image, point_1, point_2)
        intensities = np.array([line_point[2] for line_point in line_iterator])
        return np.mean(intensities)

    @staticmethod
    def compute_hisogram_of_avg_pixel_values(values):
        """
            Computes the average value from an array of pixel intensities.

            Parameters:
            -----------
            values : np.ndarray
                An array of pixel intensity values.

            Returns:
            --------
            float
                The average intensity value.
        """
        bins = 255 // 15
        values = np.nan_to_num(values, 0)
        return np.average(values)

    def filter_lines_by_intensity(self, grey: np.ndarray, hough_lines_filtered_by_orientation: List) -> np.ndarray:
        """
            Filters Hough lines based on the average intensity of pixels along each line.

            Parameters:
            -----------
            grey : np.ndarray
                The grayscale image from which lines are being filtered.
            hough_lines_filtered_by_orientation : List
                A list of Hough lines previously filtered by orientation.

            Returns:
            --------
            np.ndarray
                An array of Hough lines filtered by intensity.
        """
        _, image_thresholded = cv2.threshold(grey, self.cv_thresh, self.cv_max_val, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.float32) / 25
        img_thresholded_smoothed = cv2.filter2D(image_thresholded, -1, kernel)
        avg_intensity_of_lines = []
        for line in hough_lines_filtered_by_orientation:
            for rho, theta in line:
                point_1, point_2 = self._get_coordinates_of_hough_line(rho, theta)
                avg_intensity_of_line = self._get_avg_intensities(img_thresholded_smoothed, point_1, point_2)
                avg_intensity_of_lines.append(avg_intensity_of_line)
                for i in range(-2, 3):
                    offset = i
                    if offset == 0:
                        continue
                    parallel_point_1, parallel_point_2 = self._get_parallel_line(point_1, point_2, offset)
                    avg_intensity_of_line = self._get_avg_intensities(
                        img_thresholded_smoothed,
                        parallel_point_1,
                        parallel_point_2
                    )
                    avg_intensity_of_lines.append(avg_intensity_of_line)
        avg_intensity_of_lines = np.array(avg_intensity_of_lines)
        lines_hough_threshold = self.compute_hisogram_of_avg_pixel_values(avg_intensity_of_lines)
        hough_lines_filtered_by_intensity = []
        for line in hough_lines_filtered_by_orientation:
            for rho, theta in line:
                point_1, point_2 = self._get_coordinates_of_hough_line(rho, theta)
                for i in range(-2, 3):
                    parallel_point_1, parallel_point_2 = self._get_parallel_line(point_1, point_2, i)
                    avg_intensity_of_line = self._get_avg_intensities(
                        img_thresholded_smoothed,
                        parallel_point_1,
                        parallel_point_2
                    )
                    if avg_intensity_of_line < int(lines_hough_threshold):
                        hough_lines_filtered_by_intensity.append(np.array([[rho, theta]]))
                        break
        return np.array(hough_lines_filtered_by_intensity)
