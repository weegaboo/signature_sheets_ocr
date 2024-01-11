import cv2

import numpy as np
import sklearn.cluster as cluster

from typing import List


class GroupPoints:
    """
        A class for clustering points into groups using K-means clustering algorithm.
        It is particularly useful for grouping intersection points in image analysis.

        Attributes:
        -----------
        n_clusters : int
            The number of clusters to form.
        random_state : int
            The seed used by the random number generator for the K-means algorithm.
        kmeans : sklearn.cluster.KMeans
            The K-means clustering model.
        cv_max_val : int
            The maximum value to use in certain OpenCV operations, typically set to 255 for binary images.

        Methods:
        --------
        run(intersections: List[dict]) -> List[List[float]]:
            Clusters the provided intersection points and returns the coordinates of the cluster centers.

        plot(image: np.ndarray) -> np.ndarray:
            Draws the cluster centers on a copy of the input image and returns this modified image.
    """
    def __init__(self,
                 n_clusters: int = 56,
                 random_state: int = 42,
                 cv_max_val: int = 255):
        """
            Initializes the GroupPoints class with specific parameters for K-means clustering.

            Parameters:
            -----------
            n_clusters : int, optional
                The number of clusters to form (default is 56).
            random_state : int, optional
                The seed for the random number generator (default is 42).
            cv_max_val : int, optional
                The maximum value for binary image thresholding (default is 255).
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cv_max_val = cv_max_val


    def run(self, intersections: List) -> List:
        """
            Applies K-means clustering to the provided intersection points and returns the cluster centers.

            Parameters:
            -----------
            intersections : List[dict]
                A list of dictionaries, each containing 'x' and 'y' coordinates of an intersection point.

            Returns:
            --------
            List[List[float]]
                A list of coordinates for the cluster centers.
        """
        X = np.array([[item['x'], item['y']] for item in intersections])
        _ = self.kmeans.fit_predict(X)
        return self.kmeans.cluster_centers_.tolist()


    def plot(self, image: np.ndarray):
        """
            Draws the cluster centers as circles on a copy of the provided image.

            Parameters:
            -----------
            image : np.ndarray
                The original image on which cluster centers will be plotted.

            Returns:
            --------
            np.ndarray
                The image with drawn cluster centers.
        """
        plot_image = image.copy()
        coords = self.kmeans.cluster_centers_.tolist()
        preds = list(range(self.n_clusters))
        data = dict(zip(preds, coords))
        for point in data.values():
            x, y = int(point[0]), int(point[1])
            cv2.circle(plot_image, (x, y), 5, (0, self.cv_max_val, 0), -1)
        return plot_image
