import cv2

import numpy as np

from cv2 import Mat
from dataclasses import dataclass
from typing import List, Tuple, Any
from numpy import ndarray, dtype, generic
from .hough_lines import HoughLines


@dataclass
class KernelSizeData:
    horisontal_vertical: int = 60
    horisontal: int = 100
    vertical: int = 100


@dataclass
class HoughLinesData:
    theta: float = np.pi / 180
    threshold_1: int = 50
    threshold_2: int = 150
    aperture_size: int = 3
    threshold: int = 120


@dataclass
class VerticalLinesData:
    lower: float = (np.pi / 180) * 170
    upper: float = (np.pi / 180) * 10


@dataclass
class HorizontalLinesData:
    lower: float = (np.pi / 180) * 80
    upper: float = (np.pi / 180) * 100


class TableHoughLines(HoughLines):
    """
        A subclass of HoughLines specialized in detecting Hough lines in table images.
        It includes methods for morphological transformations, post-processing, and filtering of Hough lines.

        Attributes:
        -----------
        kernel_sizes : KernelSizeData
            Data class containing kernel sizes for horizontal and vertical line detection.
        hough_lines : HoughLinesData
            Data class containing parameters for Hough line detection.
        vertical_thresh : VerticalLinesData
            Data class containing thresholds for vertical line detection.
        horizontal_thresh : HorizontalLinesData
            Data class containing thresholds for horizontal line detection.
        postprocess_kernel_size : Tuple[int]
            The kernel size for post-processing operations.
        morphological_erode_iterations : int
            The number of iterations for erosion in morphological transformations.
        postprocess_erode_iterations : int
            The number of iterations for erosion in post-processing.
        cv_thresh : int
            The threshold value for binary thresholding in image processing.
        cv_max_val : int
            The maximum value to use in the binary thresholding operation.

        Methods:
        --------
        __init__(...):
            Initializes the TableHoughLines class with various parameters for Hough line detection and processing.

        _morphological_transformations(black_white, kernel_size):
            Applies morphological transformations to the given binary image using the specified kernel size.

        _morphological_postprocess(grey, lines):
            Applies post-processing operations to the detected lines and combines them with the original image.

        detect_horisontal_vertical(grey, black_white):
            Detects both horizontal and vertical lines in the given image.

        horizontal_detect(grey, black_white):
            Detects horizontal lines in the given image.

        vertical_detect(grey, black_white):
            Detects vertical lines in the given image.

        get_hough_lines(image):
            Detects Hough lines in the given image.

        filter_hough_lines(hough_lines):
            Filters Hough lines based on predefined angular thresholds.

        draw_hough_lines(img, hough_lines):
            Draws Hough lines on the given image.
    """
    def __init__(self,
                 kernel_sizes: KernelSizeData = KernelSizeData(),
                 hough_lines: HoughLinesData = HoughLinesData(),
                 vertical_thresh: VerticalLinesData = VerticalLinesData(),
                 horizontal_thresh: HorizontalLinesData = HorizontalLinesData(),
                 postprocess_kernel_size: Tuple[int] = (2, 2),
                 morphological_erode_iterations: int = 3,
                 postprocess_erode_iterations: int = 2,
                 line_coef: int = 1000,
                 cv_thresh: int = 128,
                 cv_max_val: int = 255):
        """
            Initializes the TableHoughLines class with specific parameters for detecting and processing Hough lines
            in table images.
        """
        super().__init__(line_coef)
        self.kernel_sizes = kernel_sizes
        self.hough_lines = hough_lines
        self.vertical_thresh = vertical_thresh
        self.horizontal_thresh = horizontal_thresh
        self.postprocess_kernel_size = postprocess_kernel_size
        self.morphological_erode_iterations = morphological_erode_iterations
        self.postprocess_erode_iterations = postprocess_erode_iterations
        self.cv_thresh = cv_thresh
        self.cv_max_val = cv_max_val

    def _morphological_transformations(self, black_white: np.ndarray, kernel_size: Tuple) -> np.ndarray:
        """
            Applies morphological transformations to enhance line features in a binary image.

            Parameters:
            -----------
            black_white : np.ndarray
                The binary image on which to perform morphological operations.
            kernel_size : Tuple
                The size of the structuring element used for morphological operations.

            Returns:
            --------
            np.ndarray
                The image after applying morphological transformations.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        detected = cv2.erode(black_white, kernel, iterations=self.morphological_erode_iterations)
        return cv2.dilate(
            detected, kernel, iterations=self.morphological_erode_iterations
        )

    def _morphological_postprocess(self, grey: np.ndarray, lines: np.ndarray) -> Tuple[
            Mat | ndarray | ndarray[Any, dtype[generic]], Mat | ndarray | ndarray[Any, dtype[generic]]]:
        """
            Applies post-processing to merge detected lines with the original grayscale image.

            Parameters:
            -----------
            grey : np.ndarray
                The original grayscale image.
            lines : np.ndarray
                The binary image with detected lines.

            Returns:
            --------
            Tuple[np.ndarray, np.ndarray]
                A tuple containing the processed lines and the inverse of the merged image.
        """
        final = cv2.getStructuringElement(cv2.MORPH_RECT, self.postprocess_kernel_size)
        lines = cv2.erode(~lines, final, iterations=self.postprocess_erode_iterations)
        thresh, lines = cv2.threshold(lines, self.cv_thresh, self.cv_max_val, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        convert_xor = cv2.bitwise_xor(grey, lines)
        inverse = cv2.bitwise_not(convert_xor)
        return lines, inverse

    def detect_horisontal_vertical(self, grey: np.ndarray, black_white: np.ndarray) -> Tuple[Any, Any]:
        """
            Detects both horizontal and vertical lines in an image using morphological transformations.

            Parameters:
            -----------
            grey : np.ndarray
                The grayscale image.
            black_white : np.ndarray
                The binary image for line detection.

            Returns:
            --------
            Tuple[np.ndarray, np.ndarray]
                A tuple containing the processed image with detected lines and the inverse of the merged image.
        """
        # kernel
        kernel_size_element = np.array(grey).shape[1] // self.kernel_sizes.horisontal_vertical
        # horizontal
        horizontal_kernel_size = (kernel_size_element, 1)
        horizontal_lines = self._morphological_transformations(black_white, horizontal_kernel_size)
        # vertical
        vertical_kernel_size = (1, kernel_size_element)
        vertical_lines = self._morphological_transformations(black_white, vertical_kernel_size)
        # combine
        combine = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        # final preprocessing
        return self._morphological_postprocess(grey, combine)

    def horizontal_detect(self, grey: np.ndarray, black_white: np.ndarray) -> Tuple[Any, Any]:
        """
            Detects horizontal lines in an image.

            Parameters:
            -----------
            grey : np.ndarray
                The grayscale image.
            black_white : np.ndarray
                The binary image for horizontal line detection.

            Returns:
            --------
            Tuple[np.ndarray, np.ndarray]
                A tuple containing the processed image with detected horizontal lines and the inverse of the merged image.
        """
        # kernel
        kernel_size_element = np.array(grey).shape[1] // self.kernel_sizes.horisontal
        # horizontal
        kernel_size = (kernel_size_element, 1)
        lines = self._morphological_transformations(black_white, kernel_size)
        return self._morphological_postprocess(grey, lines)

    def vertical_detect(self, grey: np.ndarray, black_white: np.ndarray) -> Tuple[Any, Any]:
        """
            Detects vertical lines in an image.

            Parameters:
            -----------
            grey : np.ndarray
                The grayscale image.
            black_white : np.ndarray
                The binary image for vertical line detection.

            Returns:
            --------
            Tuple[np.ndarray, np.ndarray]
                A tuple containing the processed image with detected vertical lines and the inverse of the merged image.
        """
        # kernel
        kernel_size_element = np.array(grey).shape[1] // self.kernel_sizes.vertical
        # vertical
        kernel_size = (1, kernel_size_element)
        lines = self._morphological_transformations(black_white, kernel_size)
        return self._morphological_postprocess(grey, lines)


    def get_hough_lines(self, image: np.ndarray) -> np.ndarray:
        """
            Detects Hough lines in the given image.

            Parameters:
            -----------
            image : np.ndarray
                The image in which to detect Hough lines.

            Returns:
            --------
            np.ndarray
                An array of detected Hough lines.
        """
        image = image.astype(np.uint8)
        edges = cv2.Canny(
            image, self.hough_lines.threshold_1,
            self.hough_lines.threshold_2, apertureSize=self.hough_lines.aperture_size
        )
        return cv2.HoughLines(edges, 1, self.hough_lines.theta, threshold=self.hough_lines.threshold)

    def filter_hough_lines(self, hough_lines: np.ndarray) -> List:
        """
            Filters Hough lines based on angular thresholds to separate horizontal and vertical lines.

            Parameters:
            -----------
            hough_lines : np.ndarray
                An array of detected Hough lines.

            Returns:
            --------
            List
                A list of filtered Hough lines.
        """
        hough_lines_filtered = []
        for line in hough_lines:
            for rho, theta in line:
                # vertical lines
                if self.vertical_thresh.lower < theta or theta < self.vertical_thresh.upper:
                    hough_lines_filtered.append(line)
                # horizontal lines
                elif self.horizontal_thresh.lower < theta or theta < self.horizontal_thresh.upper:
                    hough_lines_filtered.append(line)
                else:
                    print('miss_line')
        return hough_lines_filtered

    def draw_hough_lines(self, img, hough_lines):
        """
            Draws the detected Hough lines on the given image.

            Parameters:
            -----------
            img : np.ndarray
                The image on which to draw the lines.
            hough_lines : np.ndarray
                An array of Hough lines to be drawn.

            Returns:
            --------
            np.ndarray
                The image with drawn Hough lines.
        """
        img_hough_lines = img.copy()
        if hough_lines is not None:
            for line in hough_lines:
                for rho, theta in line:
                    point_1, point_2 = self._get_coordinates_of_hough_line(rho, theta)
                    cv2.line(
                        img_hough_lines,
                        (point_1.x, point_1.y),
                        (point_2.x, point_2.y),
                        (0, 0, self.cv_max_val),
                        2
                    )
        return img_hough_lines
