import numpy as np

from typing import List, Dict
from point import Point


class Cell:
    """
        A class representing a cell in a table.

        Attributes:
        -----------
        i : int
            The row index of the cell in the table.
        j : int
            The column index of the cell in the table.
        top_left : Point
            The top left corner point of the cell.
        top_right : Point
            The top right corner point of the cell.
        bottom_left : Point
            The bottom left corner point of the cell.
        bottom_right : Point
            The bottom right corner point of the cell.
        image : np.ndarray
            The image of the cell content.

        Methods:
        --------
        __init__(self, i, j, top_left, top_right, bottom_left, bottom_right, image):
            Initializes a Cell instance with row and column indices, corner points, and cell content image.
    """
    def __init__(self,
                 i: int,
                 j: int,
                 top_left: Point,
                 top_right: Point,
                 bottom_left: Point,
                 bottom_right: Point,
                 image: np.ndarray = None):
        """
            Initializes the Cell object with its position in the table and corner points.

            Parameters:
            -----------
            i : int
                The row index of the cell in the table.
            j : int
                The column index of the cell in the table.
            top_left : Point
                The top left corner point of the cell.
            top_right : Point
                The top right corner point of the cell.
            bottom_left : Point
                The bottom left corner point of the cell.
            bottom_right : Point
                The bottom right corner point of the cell.
            image : np.ndarray, optional
                The image of the cell content.
        """
        self.i = i
        self.j = j
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.image = image


class Table:
    """
        A class for representing a table in an image, consisting of cells.

        Attributes:
        -----------
        image : np.ndarray
            The image of the entire table.
        structure_points : List
            The list of points defining the structure of the table.
        n_cols : int
            The number of columns in the table.
        n_rows : int
            The number of rows in the table.
        n_structure_points : int
            The number of structural points in the table.
        data : List[List[Cell]]
            The 2D list of cells representing the table content.

        Methods:
        --------
        __init__(self, image, structure_points, n_cols, n_rows, n_structure_points):
            Initializes a Table instance with an image, structural points, and dimensions.

        split_list(list_, n) -> List:
            Splits a list into n parts of nearly equal length.

        sort_points(points, n) -> List[Dict]:
            Sorts points first by y-coordinate, then by x-coordinate within each row.

        _points2table(points) -> List[List[Cell]]:
            Converts a list of points into a 2D list of Cells representing the table.

        crop(cell) -> np.ndarray:
            Crops the part of the image corresponding to the given cell.

        __getitem__(self, index) -> Cell:
            Returns the cell at the specified row and column index.
    """
    def __init__(self,
                 image: np.ndarray,
                 structure_points: List,
                 n_cols: int = 7,
                 n_rows: int = 6,
                 n_structure_points: int = 56):
        """
            Initializes the Table object with the image of the table, structural points, and table dimensions.

            Parameters:
            -----------
            image : np.ndarray
                The image of the entire table.
            structure_points : List
                The list of points defining the structure of the table.
            n_cols : int, optional
                The number of columns in the table (default is 7).
            n_rows : int, optional
                The number of rows in the table (default is 6).
            n_structure_points : int, optional
                The number of structural points in the table (default is 56).
        """
        self.image = image
        self.structure_points = structure_points
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_structure_points = n_structure_points
        self.data = self._points2table(structure_points)

    @staticmethod
    def split_list(list_: List, n: int):
        """
            Splits a list into n parts of nearly equal length.

            Parameters:
            -----------
            list_ : List
                The list to be split.
            n : int
                The number of parts to split the list into.

            Returns:
            --------
            List
                A list of sublists, each being a part of the original list.
        """
        chunk_size = len(list_) // n
        return [list_[i:i + chunk_size] for i in range(0, len(list_), chunk_size)]

    def sort_points(self, points: List[Dict], n: int):
        """
            Sorts a list of points first by their y-coordinate, then by their x-coordinate within each row.

            Parameters:
            -----------
            points : List[Dict]
                A list of dictionaries, each representing a point with 'x' and 'y' coordinates.
            n : int
                The number of rows to sort points into.

            Returns:
            --------
            List[Dict]
                A list of points sorted first by y-coordinate and then by x-coordinate.
        """
        sorted_by_y_points = sorted(points, key=lambda point: point['y'])
        splited = self.split_list(sorted_by_y_points, n)
        result = []
        for row in splited:
            sorted_by_x_row = sorted(row, key=lambda point: point['x'])
            result.extend(sorted_by_x_row)
        return result

    def _points2table(self, points: List[Dict]):
        """
            Converts a list of points into a 2D list of Cell objects representing the table.

            Parameters:
            -----------
            points : List[Dict]
                A list of points, each represented as a dictionary with 'x' and 'y' coordinates.

            Returns:
            --------
            List[List[Cell]]
                A 2D list of Cell objects representing the table.
        """
        assert len(points) == self.n_structure_points
        prep_points = [{'x': point[0], 'y': point[1]} for point in points]
        sorted_points = self.sort_points(prep_points, self.n_rows + 1)
        table_data = []
        for cell_i in range(self.n_rows * self.n_cols):
            row_i = cell_i // self.n_cols
            col_i = cell_i % self.n_cols
            cell_ind_data = {
                'top_left_i': cell_i + row_i,
                'top_right_i': cell_i + row_i + 1,
                'bottom_left_i': self.n_cols + cell_i + row_i + 1,
                'bottom_right_i': self.n_cols + cell_i + row_i + 2,
            }
            cell_data = {}
            for name, corner_point_i in cell_ind_data.items():
                point_data = sorted_points[corner_point_i]
                name = name[:-2]
                cell_data[name] = Point(
                    x=point_data['x'],
                    y=point_data['y']
                )
            cell = Cell(
                i=row_i,
                j=col_i,
                top_left=cell_data['top_left'],
                top_right=cell_data['top_right'],
                bottom_left=cell_data['bottom_left'],
                bottom_right=cell_data['bottom_right']
            )
            cell.image = self.crop(cell)
            table_data.append(cell)
        return self.split_list(table_data, self.n_rows)

    def crop(self, cell: Cell) -> np.ndarray:
        """
            Crops the part of the table image corresponding to the given cell.

            Parameters:
            -----------
            cell : Cell
                The cell to be cropped from the table image.

            Returns:
            --------
            np.ndarray
                The cropped image corresponding to the given cell.
        """
        x_min = int(min(cell.top_left.x, cell.bottom_left.x))
        x_max = int(max(cell.top_right.x, cell.bottom_right.x))
        y_min = int(min(cell.top_left.y, cell.top_right.y))
        y_max = int(max(cell.bottom_left.y, cell.bottom_right.y))
        return self.image[y_min:y_max, x_min:x_max]

    def __getitem__(self, index):
        """
            Returns the cell at the specified row and column index.

            Parameters:
            -----------
            index : tuple
                A tuple of row and column indices (i, j).

            Returns:
            --------
            Cell
                The cell located at the specified indices in the table.
        """
        i, j = index
        return self.data[i][j]
