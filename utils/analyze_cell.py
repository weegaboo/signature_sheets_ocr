import cv2
import os

import numpy as np

from typing import Tuple


class AnalyzeCell:
    """
        A class for analyzing cell areas in an image, particularly focusing on identifying
        empty cells and calculating areas based on contours and thresholding.

        Attributes:
        -----------
        x_indent : float
            Horizontal margin as a proportion of the image width to exclude from analysis.
        y_indent : float
            Vertical margin as a proportion of the image height to exclude from analysis.
        empty_threshold : float
            Threshold proportion to determine if a cell is empty based on the ratio of text area to total area.
        cv_max_val : int
            Maximum value to use with the binary thresholding operation.
        cv_thresh : int
            Threshold value for the binary thresholding operation.
        count : int
            Counter to keep track of the number of processed images.

        Methods:
        --------
        calculate_areas(image: np.ndarray) -> Tuple[int, int, int]:
            Calculates the total, filled, and empty areas of the provided image.

        calculate_contours_areas(image: np.ndarray) -> Tuple[int, int]:
            Calculates the total area of contours in the image and the total area of the image.

        is_empty(image: np.ndarray) -> bool:
            Determines whether a cell in the image is empty based on defined thresholds and area calculations.
    """
    def __init__(self,
                 x_indent: float = 0.1,
                 y_indent: float = 0.1,
                 empty_threshold: float = 0.05,
                 cv_max_val: int = 255,
                 cv_thresh: int = 200):
        self.x_indent = x_indent
        self.y_indent = y_indent
        self.empty_threshold = empty_threshold
        self.cv_max_val = cv_max_val
        self.cv_thresh = cv_thresh
        self.count = 0

    def calculate_areas(self, image: np.ndarray) -> Tuple:
        """
            Calculates and returns the total, filled, and empty areas of the image.

            Parameters:
            -----------
            image : np.ndarray
                The input image for area calculation.

            Returns:
            --------
            Tuple[int, int, int]
                A tuple containing the total area of the image, the filled area (non-zero pixels),
                and the empty area (zero pixels).
        """
        _, binary_image = cv2.threshold(image, self.cv_thresh, self.cv_max_val, cv2.THRESH_BINARY)
        empty_area = cv2.countNonZero(binary_image)
        height, width = binary_image.shape[:2]
        image_size = height * width
        filled_area = image_size - empty_area
        return image_size, filled_area, empty_area

    def calculate_contours_areas(self, image: np.ndarray):
        """
            Analyzes the image to calculate the total area covered by contours and the total area of the image.

            This method applies thresholding and contour detection to the image, saves an image with drawn contours,
            and calculates the total area covered by these contours.

            Parameters:
            -----------
            image : np.ndarray
                The input image for contour analysis.

            Returns:
            --------
            Tuple[int, int]
                A tuple containing the total area covered by contours and the total area of the image.
        """
        image = image.copy()
        _, thresholded = cv2.threshold(image, 0, self.cv_max_val, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(f'cropped_contours_{self.count}.png', image)
        total_area = image.shape[0] * image.shape[1]
        total_rectangle_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rectangle_area = w * h
            total_rectangle_area += rectangle_area
        return total_rectangle_area, total_area


    def is_empty(self, image: np.ndarray) -> bool:
        """
            Determines if the provided cell (image) is empty based on the proportion of the text area to the total area.

            The method crops the image based on defined margins, calculates the filled and total areas,
            and uses the proportion of these areas to determine if the cell is empty.

            Parameters:
            -----------
            image : np.ndarray
                The image representing the cell to be analyzed.

            Returns:
            --------
            bool
                Returns True if the cell is considered empty, otherwise False.
        """
        height, width = image.shape[:2]
        lower_y, upper_y = int(height * self.y_indent), int(height * (1 - self.y_indent))
        lower_x, upper_x = int(width * self.x_indent), int(width * (1 - self.x_indent))
        cropped = image[lower_y:upper_y, lower_x:upper_x]
        steps_path = 'process_steps/'
        cv2.imwrite(os.path.join(steps_path, f'table_data/cropped_{self.count}.png'), cropped)
        image_size, filled_area, empty_area = self.calculate_areas(cropped)
        text_area, total_area = self.calculate_contours_areas(cropped)
        self.count += 1
        return filled_area == 0 or text_area / total_area < self.empty_threshold
