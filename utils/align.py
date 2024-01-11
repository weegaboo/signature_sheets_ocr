import cv2

import numpy as np

from typing import Any

from cv2 import Mat
from numpy import ndarray, dtype, generic


class Align:
    """
        A class for aligning two images using feature matching and homography.

        Attributes:
        -----------
        cv_thresh : int
            Threshold value for the binary thresholding operation.
        cv_max_val : int
            Maximum value to use with the binary thresholding operation.
        orb_n_features : int
            The number of features to retain for the ORB feature detector.
        top_n_matches : float
            The proportion (0 to 1) of top matches to consider for homography calculation.
        matches_shape : int
            The dimension of the points in the feature matching process.

        Methods:
        --------
        alignment(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
            Aligns the given image with the reference image using ORB feature detection,
            feature matching, and homography. Returns the aligned image.

        run(image: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Mat, Any]:
            Performs alignment of the given image with the reference image and additionally
            returns the grayscale and binary inverted grayscale of the aligned image.

        Example Usage:
        --------------
            aligner = Align()
            aligned_image, gray_image, binary_image = aligner.run(image, reference_image)
    """
    def __init__(self,
                 cv_thresh: int = 128,
                 cv_max_val: int = 255,
                 orb_n_features: int = 1000,
                 top_n_matches: int = 0.9,
                 matches_shape: int = 2):
        """
            Initializes the Align class with specified parameters for image alignment.

            Parameters:
            -----------
            cv_thresh : int, optional
                Threshold value for binary thresholding (default is 128).
            cv_max_val : int, optional
                Maximum value to use with the binary thresholding operation (default is 255).
            orb_n_features : int, optional
                The number of features to be used by the ORB feature detector (default is 1000).
            top_n_matches : float, optional
                The proportion (0 to 1) of top matches to keep for homography (default is 0.9).
            matches_shape : int, optional
                The dimension of the points (e.g., 2 for 2D points) in feature matching (default is 2).
        """
        self.cv_thresh = cv_thresh
        self.cv_max_val = cv_max_val
        self.orb_n_features = orb_n_features
        self.top_n_matches = top_n_matches
        self.matches_shape = matches_shape

    def alignment(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
            Aligns an image to a reference image using feature matching and homography.

            This method converts both images to grayscale, detects key points and computes descriptors using ORB,
            matches features between the two images, sorts the matches based on Hamming distance,
            uses the top matches to compute a homography matrix, and applies this matrix to warp the input image.

            Parameters:
            -----------
            image : np.ndarray
                The image to be aligned.
            reference : np.ndarray
                The reference image to align the input image to.

            Returns:
            --------
            np.ndarray
                The aligned image that has been warped to match the perspective of the reference image.
        """
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reference_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        height, width = reference_grey.shape
        # Configure ORB feature detector Algorithm with 1000 features.
        orb_detector = cv2.ORB_create(self.orb_n_features)
        # Extract key points and descriptors for both images
        key_point_1, des_1 = orb_detector.detectAndCompute(grey, None)
        key_point_2, des_2 = orb_detector.detectAndCompute(reference_grey, None)
        # Match features between two images using Brute Force matcher with Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match the two sets of descriptors.
        matches = list(matcher.match(des_1, des_2))
        # Sort matches on the basis of their Hamming distance.
        matches.sort(key=lambda x: x.distance)
        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches) * self.top_n_matches)]
        no_of_matches = len(matches)
        # Define 2x2 empty matrices
        p1 = np.zeros((no_of_matches, self.matches_shape))
        p2 = np.zeros((no_of_matches, self.matches_shape))
        # Storing values to the matrices
        for i in range(len(matches)):
            p1[i, :] = key_point_1[matches[i].queryIdx].pt
            p2[i, :] = key_point_2[matches[i].trainIdx].pt
        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        return cv2.warpPerspective(image, homography, (width, height))

    def run(self, image: np.ndarray, reference: np.ndarray) -> tuple[
            np.ndarray, Mat | ndarray | ndarray[Any, dtype[generic]], Any]:
        """
            Executes the alignment process on the given image with the reference image and
            returns the aligned image along with its grayscale and binary inverted grayscale versions.

            This method first aligns the input image with the reference image.
            Then, it converts the aligned image to grayscale and creates a binary inverted grayscale image.

            Parameters:
            -----------
            image : np.ndarray
                The image to be processed.
            reference : np.ndarray
                The reference image for alignment.

            Returns:
            --------
            tuple[np.ndarray, Mat, Any]
                A tuple containing the aligned image, its grayscale version,
                and its binary inverted grayscale version.
        """
        aligned = self.alignment(image, reference)
        grey = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        black_white = 255 - grey
        return aligned, grey, black_white
