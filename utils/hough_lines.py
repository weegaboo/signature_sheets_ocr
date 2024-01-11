import numpy as np

from point import Point
from typing import Tuple


class HoughLines:
    """
        A class for handling operations related to Hough Lines in image processing.

        This class primarily deals with converting the polar coordinates (rho, theta)
        obtained from a Hough Transform to Cartesian coordinates, representing the endpoints
        of the detected lines in the image.

        Attributes:
        -----------
        line_coef : int
            A coefficient used to extend the line for visualization purposes.

        Methods:
        --------
        _get_coordinates_of_hough_line(rho: float, theta: float) -> Tuple[Point, Point]:
            Converts polar coordinates (rho, theta) to Cartesian coordinates and returns
            the endpoints of the corresponding line segment.
    """
    def __init__(self, line_coef: int = 1000):
        """
            Initializes the HoughLines class with a specified line coefficient.

            Parameters:
            -----------
            line_coef : int, optional
                A coefficient to determine the length of the line segments for visualization (default is 1000).
        """
        self.line_coef = line_coef

    def _get_coordinates_of_hough_line(self, rho, theta) -> Tuple[Point, Point]:
        """
            Converts a line from polar coordinates (rho, theta) to Cartesian coordinates and
            returns the endpoints of the line segment.

            This method uses the mathematical relationships between polar and Cartesian coordinates
            to calculate the line's endpoints, extending the line by a factor of 'line_coef'.

            Parameters:
            -----------
            rho : float
                The distance from the origin to the line along a perpendicular from the origin.
            theta : float
                The angle formed by this perpendicular line and the horizontal axis.

            Returns:
            --------
            Tuple[Point, Point]
                A tuple containing two Point objects representing the endpoints of the line segment.
        """
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho  # TODO: use pol2cart
        y0 = b * rho  # TODO: use pol2cart
        x1 = int(x0 + self.line_coef * (-b))
        y1 = int(y0 + self.line_coef * (a))
        x2 = int(x0 - self.line_coef * (-b))
        y2 = int(y0 - self.line_coef * (a))
        point_1 = Point(x1, y1)
        point_2 = Point(x2, y2)
        return point_1, point_2
