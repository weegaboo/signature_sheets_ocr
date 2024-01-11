import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster

from .hough_lines import HoughLines
from point import Point
from typing import List

class Merge(HoughLines):
    """
        A subclass of HoughLines, designed for merging similar lines detected via the Hough Transform
        using hierarchical clustering based on the Hausdorff distance between lines.

        Inherits the line coefficient calculation functionality from HoughLines and extends it
        to include the merging of similar lines.

        Attributes:
        -----------
        cv_max_val : int
            The maximum value used in certain OpenCV operations, typically set to 255 for binary images.
        max_distance : int
            The maximum distance threshold for merging lines in the hierarchical clustering algorithm.

        Methods:
        --------
        get_hausdorff_distance(line_a: List[Point], line_b: List[Point]) -> float:
            Calculates the Hausdorff distance between two lines, each represented as a list of Points.

        polar2cartesian(hough_lines_segmented: List[List[Tuple[float, float]]]) -> Tuple[List[List[Point]], List[List[Point]]]:
            Converts polar coordinates of Hough lines to Cartesian coordinates.

        run(hough_lines_cartesian: List[List[Point]], direction: str) -> List[List[Point]]:
            Merges similar lines using hierarchical clustering and sorts them based on their orientation.
    """
    def __init__(self,
                 line_coef: int = 1000,
                 cv_max_val: int = 255,
                 max_distance: int = 30):
        """
            Initializes the Merge class with specific parameters for line merging.

            Parameters:
            -----------
            line_coef : int
                A coefficient to determine the length of the line segments for visualization (inherited).
            cv_max_val : int
                The maximum value for binary image thresholding.
            max_distance : int
                The maximum distance threshold for merging lines in clustering.
        """
        super().__init__(line_coef)
        self.cv_max_val = cv_max_val
        self.max_distance = max_distance

    @staticmethod
    def get_hausdorff_distance(line_a: List[Point], line_b: List[Point]):
        """
            Computes the Hausdorff distance between two lines for comparison.

            Parameters:
            -----------
            line_a : List[Point]
                The first line, represented as a list of two Points.
            line_b : List[Point]
                The second line, represented as a list of two Points.

            Returns:
            --------
            float
                The Hausdorff distance between the two lines.
        """
        line_a = list(map(lambda point: [point.x, point.y], line_a))
        line_b = list(map(lambda point: [point.x, point.y], line_b))
        return ssd.directed_hausdorff(line_a, line_b)[0]

    def polar2cartesian(self, hough_lines_segmented: List):
        """
            Converts the polar coordinates of Hough lines to Cartesian coordinates.

            Parameters:
            -----------
            hough_lines_segmented : List[List[Tuple[float, float]]]
                A list containing lists of Hough lines in polar coordinates.

            Returns:
            --------
            Tuple[List[List[Point]], List[List[Point]]]
                A tuple containing two lists of Hough lines in Cartesian coordinates,
                separated as horizontal and vertical lines.
        """
        hough_lines_horizontal = hough_lines_segmented[0]
        hough_lines_vertical = hough_lines_segmented[1]
        hough_lines_horizontal_cartesian = []
        hough_lines_vertical_cartesian = []
        for line in hough_lines_horizontal:
            for rho, theta in line:
                point_1, point_2 = self._get_coordinates_of_hough_line(rho, theta)
                point_2.x += 2000
                line_points = [point_1, point_2]
                if point_1.x > point_2.x:
                    line_points[0], line_points[1] = line_points[1], line_points[0]
                hough_lines_horizontal_cartesian.append(line_points)
        for line in hough_lines_vertical:
            for rho, theta in line:
                point_1, point_2 = self._get_coordinates_of_hough_line(rho, theta)
                line_points = [point_1, point_2]
                if point_1.y > point_2.y:
                    line_points[0], line_points[1] = line_points[1], line_points[0]
                hough_lines_vertical_cartesian.append(line_points)
        return (hough_lines_horizontal_cartesian,
                hough_lines_vertical_cartesian)

    def run(self, hough_lines_cartesian: List, direction: str) -> List:
        """
            Merges similar lines based on the Hausdorff distance and hierarchical clustering,
            and sorts them based on their orientation.

            Parameters:
            -----------
            hough_lines_cartesian : List[List[Point]]
                A list of lines, each represented as a list of Points in Cartesian coordinates.
            direction : str
                The direction of lines to be merged ('vertical' or 'horizontal').

            Returns:
            --------
            List[List[Point]]
                A list of merged and sorted lines.
        """
        distance_matrix = []
        for i in range(len(hough_lines_cartesian)):
            distance_matrix.append([])
            for j in range(len(hough_lines_cartesian)):
                line_i = hough_lines_cartesian[i]
                line_j = hough_lines_cartesian[j]
                distance = self.get_hausdorff_distance(line_i, line_j)
                distance_matrix[i].append(distance)
        distance_matrix_squared = ssd.squareform(distance_matrix)
        Z = hcluster.linkage(distance_matrix_squared)
        clusters = hcluster.fcluster(Z, self.max_distance, criterion='distance')
        number_of_clusters = len(set(clusters))
        clusters_grouped = np.empty((number_of_clusters,), dtype=object)
        for i in range(len(clusters)):
            cluster_id = clusters[i] - 1
            if clusters_grouped[cluster_id] is None:
                clusters_grouped[cluster_id] = []
            clusters_grouped[cluster_id].append(i)
        hough_lines_merged_indices = [
            cluster_grouped[0] for cluster_grouped in clusters_grouped
        ]
        hough_lines_merged = []
        for hough_line_index in hough_lines_merged_indices:
            line_points = hough_lines_cartesian[hough_line_index]
            hough_lines_merged.append(line_points)
        if direction == 'horizontal':
            index = 1
        elif direction == 'vertical':
            index = 0
        else:
            raise Exception('direction!')
        return sorted(hough_lines_merged, key=lambda line: line[0].y)
