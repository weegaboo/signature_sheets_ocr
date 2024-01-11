import os
import cv2

import numpy as np

from typing import List
from .align import Align
from .table_detection import TableDetector
from .table_image_to_hough_lines import TableHoughLines
from .preprocess_hough_lines import PreprocessHough
from .segment_hough_lines import Segmenter
from .merge_hough_lines import Merge
from .intersections import Intersector
from .intersections_to_table_points import GroupPoints
from .structure_points_to_table_object import Table, Cell
from .analyze_cell import AnalyzeCell


class TablePipe:
    """
        A pipeline class for processing table images. It integrates multiple steps
        such as alignment, table detection, Hough line processing, intersections finding,
        and cell analysis to identify empty cells in a table image.

        Attributes:
        -----------
        align : Align
            An instance of the Align class for image alignment.
        table_detector : TableDetector
            An instance of the TableDetector class for detecting tables in images.
        hough : TableHoughLines
            An instance of the TableHoughLines class for detecting Hough lines.
        preproc_hough : PreprocessHough
            An instance of the PreprocessHough class for preprocessing Hough lines.
        segmenter : Segmenter
            An instance of the Segmenter class for segmenting Hough lines.
        merge : Merge
            An instance of the Merge class for merging Hough lines.
        intersector : Intersector
            An instance of the Intersector class for finding line intersections.
        grouper : GroupPoints
            An instance of the GroupPoints class for clustering intersection points.
        cell_analyzer : AnalyzeCell
            An instance of the AnalyzeCell class for analyzing cells in a table.
        table : Table
            An instance of the Table class representing the processed table.
        empty_cells : List[Cell]
            A list of Cell instances that are identified as empty.

        Methods:
        --------
        plot_empty(image: np.ndarray, cells: List[Cell]) -> np.ndarray:
            Overlays rectangles on empty cells in the image.

        run(image: np.ndarray, image_reference: np.ndarray) -> np.ndarray:
            Executes the entire pipeline on the given image and returns the processed image.
    """
    def __init__(self,
                 align: Align = Align(),
                 table_detector: TableDetector = TableDetector(),
                 hough: TableHoughLines = TableHoughLines(),
                 preproc_hough: PreprocessHough = PreprocessHough(),
                 segmenter: Segmenter = Segmenter(),
                 merge: Merge = Merge(),
                 intersector: Intersector = Intersector(),
                 grouper: GroupPoints = GroupPoints(),
                 cell_analyzer: AnalyzeCell = AnalyzeCell()):
        """
            Initializes the TablePipe class with instances of all the necessary components
            for table image processing.
        """
        self.align = align
        self.table_detector = table_detector
        self.hough = hough
        self.preproc_hough = preproc_hough
        self.segmenter = segmenter
        self.merge = merge
        self.intersector = intersector
        self.grouper = grouper
        self.cell_analyzer = cell_analyzer
        self.table = None
        self.empty_cells = []


    @staticmethod
    def plot_empty(image: np.ndarray, cells: List[Cell]):
        """
            Draws semi-transparent rectangles over empty cells in a table image.

            Parameters:
            -----------
            image : np.ndarray
                The original image of the table.
            cells : List[Cell]
                A list of Cell instances to be highlighted as empty.

            Returns:
            --------
            np.ndarray
                The modified image with highlighted empty cells.
        """
        overlay = image.copy()
        red_color = (0, 0, 255)
        for cell in cells:
            pt1 = (int(cell.top_left.x), int(cell.top_left.y))
            pt2 = (int(cell.bottom_right.x), int(cell.bottom_right.y))
            cv2.rectangle(overlay, pt1, pt2, red_color, -1)
        alpha = 0.4
        return cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)


    def run(self, image: np.ndarray, image_reference: np.ndarray):
        """
            Processes the given table image through various stages including alignment,
            table detection, Hough line processing, finding intersections, and cell analysis.

            Parameters:
            -----------
            image : np.ndarray
                The original image containing a table.
            image_reference : np.ndarray
                A reference image used for alignment purposes.

            Returns:
            --------
            np.ndarray
                The processed image with identified empty cells.
        """
        steps_path = './process_steps/'
        # Align
        aligned, grey, black_white = self.align.run(image, image_reference)
        cv2.imwrite(os.path.join(steps_path, 'align/grey_aligned.png'), grey)
        cv2.imwrite(os.path.join(steps_path, 'align/black_white.png'), black_white)
        cv2.imwrite(os.path.join(steps_path, 'align/aligned.png'), aligned)
        # Table Crop
        grey_table, table_coords = self.table_detector.crop_table_from_image(grey)
        black_white_table = 255 - grey_table
        cv2.imwrite(os.path.join(steps_path, 'table_crop/grey_table.png'), grey_table)
        cv2.imwrite(os.path.join(steps_path, 'table_crop/black_white_table.png'), black_white_table)
        # Get Hough Lines
        combine, inverse = self.hough.detect_horisontal_vertical(grey_table, black_white_table)
        grey_table_structure, _ = self.table_detector.crop_table_from_image(combine, x_up=1, y_up=1)
        black_white_table_structure = 255 - grey_table_structure
        horizontal_table, _ = self.hough.horizontal_detect(
            grey_table_structure,
            black_white_table_structure
        )
        vertical_table, _ = self.hough.vertical_detect(
            grey_table_structure,
            black_white_table_structure
        )
        # cv2.imwrite('grey_table_structure.png', grey_table_structure)
        hough_lines_horizontal = self.hough.get_hough_lines(horizontal_table)
        # print(f"hough_lines_horizontal: {len(hough_lines_horizontal)}")
        hough_lines_vertical = self.hough.get_hough_lines(vertical_table)
        # print(f"hough_lines_vertical: {len(hough_lines_vertical)}")
        hough_lines = np.concatenate([hough_lines_horizontal, hough_lines_vertical])
        hough_lines_filtered_by_orientation = self.hough.filter_hough_lines(hough_lines)
        img_hough_lines = self.hough.draw_hough_lines(
            grey_table.copy(),
            hough_lines_filtered_by_orientation
        )
        cv2.imwrite(os.path.join(steps_path, 'hough_lines/img_hough_lines.png'), img_hough_lines)
        # print(f"intensity len: {len(hough_lines_filtered_by_orientation)}")
        # Preprocess Hough Lines
        hough_lines_filtered_by_intensity = self.preproc_hough.filter_lines_by_intensity(
            grey_table_structure,
            hough_lines_filtered_by_orientation
        )
        # Segment Hough
        hough_lines_segmented = self.segmenter.run(hough_lines_filtered_by_intensity)
        hough_lines_segmented_image = self.segmenter.plot(grey_table, hough_lines_segmented)
        cv2.imwrite(os.path.join(steps_path, 'hough_lines/segmented_hough.jpg'), hough_lines_segmented_image)
        # Merge Hough
        hough_lines_horizontal, hough_lines_vertical = self.merge.polar2cartesian(hough_lines_segmented)
        hough_lines_horizontal_merged = self.merge.run(hough_lines_horizontal, direction='horizontal')
        hough_lines_vertical_merged = self.merge.run(hough_lines_vertical, direction='vertical')
        # Intersections
        intersections = self.intersector.run(grey_table, hough_lines_horizontal_merged, hough_lines_vertical_merged)
        intersections_image = self.intersector.plot(grey_table, intersections)
        cv2.imwrite(os.path.join(steps_path, 'intersections.jpg'), intersections_image)
        # Table points clustering
        structure_points = self.grouper.run(intersections)
        grey_table_with_points_image = self.grouper.plot(grey_table)
        cv2.imwrite(os.path.join(steps_path, 'grey_table_with_points_image.jpg'), grey_table_with_points_image)
        # Get table object from points
        self.table = Table(image=grey_table, structure_points=structure_points)
        table_image_with_empty = cv2.cvtColor(self.table.image.copy(), cv2.COLOR_GRAY2RGB)
        for i in range(self.table.n_rows):
            for j in range(self.table.n_cols):
                cell = self.table[i, j]
                if i != 0 and j != 0 and self.cell_analyzer.is_empty(cell.image):
                    self.empty_cells.append(cell)
                cv2.imwrite(os.path.join(steps_path, f'table_data/cell_{i}_{j}.jpg'), cell.image)
        table_image_with_empty = self.plot_empty(table_image_with_empty, self.empty_cells)
        cv2.imwrite(
            os.path.join(steps_path, 'table_data/table_image_with_empty.jpg'),
            table_image_with_empty,
        )
        return table_image_with_empty

        