import cv2

import numpy as np


class TableDetector:
    """
        A class for detecting and extracting the table region from an image.

        This class uses thresholding and contour detection to identify table-like regions
        in an image and extract them.

        Attributes:
        -----------
        cv_thresh : int
            The threshold value for binary thresholding in the table detection process.
        cv_max_val : int
            The maximum value to use in the binary thresholding operation.

        Methods:
        --------
        crop_table_from_image(grey: np.ndarray, x_low: float, x_up: float, y_low: float, y_up: float):
            Detects and extracts the table region from a grayscale image.
    """
    def __init__(self,
                 cv_thresh: int = 190,
                 cv_max_val: int = 255):
        """
            Initializes the TableDetector with specific parameters for binary thresholding.

            Parameters:
            -----------
            cv_thresh : int, optional
                The threshold value for binary thresholding in the table detection process (default is 190).
            cv_max_val : int, optional
                The maximum value for binary thresholding (default is 255).
        """
        self.cv_thresh = cv_thresh
        self.cv_max_val = cv_max_val

    def crop_table_from_image(self,
                              grey: np.ndarray,
                              x_low: float = 0.6,
                              x_up: float = 0.99,
                              y_low: float = 0.4,
                              y_up: float = 0.9):
        """
            Detects and extracts the table region from a grayscale image based on specified parameters.

            The method applies binary thresholding and contour detection to identify rectangular regions
            resembling a table. It then selects the largest such region based on area.

            Parameters:
            -----------
            grey : np.ndarray
                The grayscale image from which the table region is to be extracted.
            x_low : float, optional
                The lower bound for the width ratio of detected rectangles (default is 0.6).
            x_up : float, optional
                The upper bound for the width ratio of detected rectangles (default is 0.99).
            y_low : float, optional
                The lower bound for the height ratio of detected rectangles (default is 0.4).
            y_up : float, optional
                The upper bound for the height ratio of detected rectangles (default is 0.9).

            Returns:
            --------
            Tuple[np.ndarray, dict]
                A tuple containing the cropped table image and the coordinates of the table region.
        """
        image_height, image_width = grey.shape[:2]
        ret, thresh_value = cv2.threshold(grey, self.cv_thresh, self.cv_max_val, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects_coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            delta_x = w / image_width
            delta_y = h / image_height
            area = cv2.contourArea(contour)
            if (x_low <= delta_x <= x_up) and (y_low <= delta_y <= y_up):
                table_coords = {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'area': cv2.contourArea(contour)
                }
                rects_coords.append(table_coords)
        table_coords = max(rects_coords, key=lambda item: item['area'])
        table_image = grey[
            table_coords['y']:table_coords['y']+table_coords['h'],
            table_coords['x']:table_coords['x']+table_coords['w']
        ]
        return table_image, table_coords

