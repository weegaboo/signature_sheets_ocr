class Point:
    """
        A simple class representing a point in a 2D space.

        Attributes:
        -----------
        x : float or int
            The x-coordinate of the point.
        y : float or int
            The y-coordinate of the point.

        Methods:
        --------
        __init__(self, x: float or int, y: float or int):
            Initializes a Point instance with x and y coordinates.
    """
    def __init__(self, x, y):
        """
            Initializes the Point object with specified x and y coordinates.

            Parameters:
            -----------
            x : float or int
                The x-coordinate of the point.
            y : float or int
                The y-coordinate of the point.
        """
        self.x = x
        self.y = y
