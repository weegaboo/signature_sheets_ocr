import cv2

import numpy as np

from point import Point
from typing import Tuple, List
from shapely.geometry import LineString


class Intersector:
    """
        A class for finding intersections between lines, particularly useful in image analysis
        for identifying points where horizontal and vertical lines intersect.

        Attributes:
        -----------
        cv_max_val : int
            The maximum value to use in certain OpenCV operations, typically set to 255 for binary images.

        Methods:
        --------
        _get_intersection(line_a: List[Point], line_b: List[Point]) -> Tuple[int, int]:
            Returns the intersection point of two lines, each represented by a list of two Points.

        get_segment_intersections(lines: List[List[Point]], image_width: int, image_height: int) -> List[dict]:
            Finds and returns intersections among a list of line segments within specified image dimensions.

        run(grey: np.ndarray, hough_lines_horizontal_merged: List, hough_lines_vertical_merged: List) -> List[dict]:
            Processes a grayscale image and lists of horizontal and vertical lines to find intersections.

        plot(image: np.ndarray, intersections: List[dict]) -> np.ndarray:
            Draws the found intersections on a copy of the input image and returns this modified image.
    """
    def __init__(self, cv_max_val: int = 255):
        """
            Initializes the Intersector class with a maximum value for binary image thresholding.

            Parameters:
            -----------
            cv_max_val : int, optional
                The maximum value to use for binary thresholding (default is 255).
        """
        self.cv_max_val = cv_max_val

    @staticmethod
    def _get_intersection(line_a: List[Point], line_b: List[Point]) -> Tuple:
        """
            Computes the intersection point of two lines, each defined by two Points.

            Parameters:
            -----------
            line_a : List[Point]
                The first line, represented as a list of two Points.
            line_b : List[Point]
                The second line, represented as a list of two Points.

            Returns:
            --------
            Tuple[int, int]
                The coordinates of the intersection point as a tuple (x, y).
                Returns an empty tuple if there is no intersection.
        """
        line_a = list(map(lambda point: [point.x, point.y], line_a))
        line_b = list(map(lambda point: [point.x, point.y], line_b))
        line1 = LineString([line_a[0], line_a[1]])
        line2 = LineString([line_b[0], line_b[1]])
        int_pt = line1.intersection(line2)
        return (round(int_pt.x), round(int_pt.y)) if list(int_pt.coords) else ()

    def get_segment_intersections(self, lines: List, image_width: int, image_height: int) -> List:
        """
            Identifies intersection points among a list of line segments within an image's dimensions.

            Parameters:
            -----------
            lines : List[List[Point]]
                A list of lines, where each line is represented as a list of Points.
            image_width : int
                The width of the image in pixels.
            image_height : int
                The height of the image in pixels.

            Returns:
            --------
            List[dict]
                A list of dictionaries, each representing an intersection with its coordinates and indices.
        """
        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i + 1:]:
                for index_horizontal, line1 in enumerate(group):
                    for index_vertical, line2 in enumerate(next_group):
                        intersection_point = self._get_intersection(line1, line2)
                        if intersection_point == []:
                            continue
                        x = intersection_point[0]
                        if x > image_width:
                            x = image_width
                        elif x < 0:
                            x = 0
                        y = intersection_point[1]
                        if y > image_height:
                            y = image_height
                        elif y < 0:
                            y = 0
                        intersection = {
                            'id': len(intersections),
                            'x': x,
                            'y': y,
                            'horizontal_index': index_horizontal,
                            'vertical_index': index_vertical
                        }
                        intersections.append(intersection)
        return intersections

    def run(self, grey: np.ndarray, hough_lines_horizontal_merged: List, hough_lines_vertical_merged: List) -> List:
        """
            Processes a grayscale image to find intersections between provided horizontal and vertical lines.

            Parameters:
            -----------
            grey : np.ndarray
                The grayscale image to analyze.
            hough_lines_horizontal_merged : List[List[Point]]
                A list of horizontal lines, each represented by two Points.
            hough_lines_vertical_merged : List[List[Point]]
                A list of vertical lines, each represented by two Points.

            Returns:
            --------
            List[dict]
                A list of dictionaries detailing the intersections found in the image.
        """
        hough_lines_segmented_merged = [hough_lines_horizontal_merged, hough_lines_vertical_merged]
        image_height, image_width = grey.shape[:2]
        return self.get_segment_intersections(
            hough_lines_segmented_merged, image_width, image_height
        )

    def plot(self, image: np.ndarray, intersections: List):
        """
            Draws circles at intersection points on a copy of the input image.

            Parameters:
            -----------
            image : np.ndarray
                The original image on which intersections will be plotted.
            intersections : List[dict]
                A list of dictionaries containing intersection data.

            Returns:
            --------
            np.ndarray
                The image with drawn intersections.
        """
        intersections_img = image.copy()
        x_intersections = [item['x'] for item in intersections]
        y_intersections = [item['y'] for item in intersections]
        for i in range(len(intersections)):
            cv2.circle(intersections_img, (x_intersections[i], y_intersections[i]), 5, (0, self.cv_max_val, 0), 2)
        return intersections_img
