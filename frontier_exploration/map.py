import numpy as np
import math 

class Frontier():
    def __init__(self, size: int = 1, min_distance : float = float('inf')):
        self.size = size
        self.min_distance = min_distance
        self.travel_point = None # location the agent should travel to
        self.points = []
    def __str__(self):
        return "Frontier(size=" + str(self.size) + ", min_distance=" + str(self.min_distance) + ", travel_point=" + str(self.travel_point) + ", points=" + str([str(x) for x in self.points]) + ")"
    
class Point():
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
        
    def copy(self):
        return Point(self.x, self.y)
    
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
    
class Map():
    def __init__(self, step_ego_grid_crops_3: np.ndarray):
        """
        Convert probabilities to label (0, 1, 2)
        If max probability < 0.5, then default to VOID (0)
        """
        unknown = np.max(step_ego_grid_crops_3, 0) < 0.4
        step_labels = np.argmax(step_ego_grid_crops_3, 0) * np.logical_not(unknown)

        self.map = step_labels
        self.size_y, self.size_x = self.map.shape
        self.proj_grid = step_ego_grid_crops_3

    def getSizeInCells(self):
        return self.size_x, self.size_y

    def getCharMap(self):
        return self.map.flatten()

    def center(self):
        return self.size_x // 2, self.size_y // 2

    def getIndex(self, mx: int, my: int):
        return my * self.size_x + mx
    
    def indexToPoint(self, index: int):
        my = index // self.size_x
        mx = index - (my * self.size_x)
        return Point(mx, my)

    
    
    """
     * @brief Determine 4-connected neighbourhood of an input cell, checking for map edges
     * @param idx input cell index
     * @param costmap Reference to map data
     * @return neighbour cell indexes
    """
    def nhood4(self, idx: int):
        # get 4-connected neighbourhood indexes, check for edge of map
        out = []
    
        if (idx > self.size_x * self.size_y - 1):
            raise Exception("Evaluating nhood for offmap point")
            return out
        
        if (idx % self.size_x > 0):
            out.append(idx - 1)
        
        if (idx % self.size_x < self.size_x - 1):
            out.append(idx + 1)
        
        if (idx >= self.size_x):
            out.append(idx - self.size_x)
        
        if (idx < self.size_x * (self.size_y - 1)):
            out.append(idx + self.size_x)
        
        return out
    
    
    """
     * @brief Determine 8-connected neighbourhood of an input cell, checking for map edges
     * @param idx input cell index
     * @param costmap Reference to map data
     * @return neighbour cell indexes
    """
    def nhood8(self, idx: int):
        # get 8-connected neighbourhood indexes, check for edge of map
        out = self.nhood4(idx)
        
        if (idx > self.size_x * self.size_y - 1):
            return out
        
        if (idx % self.size_x > 0 and idx >= self.size_x):
            out.append(idx - 1 - self.size_x)
        
        if (idx % self.size_x > 0 and idx < self.size_x * (self.size_y-1)):
            out.append(idx - 1 + self.size_x)
    
        if (idx % self.size_x < self.size_x - 1 and idx >= self.size_x):
            out.append(idx + 1 - self.size_x)
    
        if (idx % self.size_x < self.size_x - 1 and idx < self.size_x * (self.size_y-1)):
            out.append(idx + 1 + self.size_x)
    
        return out
    
    """
     * @brief Find nearest cell of a specified value
     * @param start Index initial cell to search from
     * @param val Specified value to search for
     * @param costmap Reference to map data
     * @return (bool, int): True if a cell with the requested value was found, False otherwise
                            int Index of located cell, None if not found
    """
    def nearestCell(self, start: int, val: int):
        flatMap = self.getCharMap()
    
        if start >= self.size_x * self.size_y:
            return False, None
        
        # initialize breadth first search
        bfs = []
        visited_flag = [False] * (self.size_x * self.size_y)
    
        # push initial cell
        bfs.append(start)
        visited_flag[start] = True
    
        # search for neighbouring cell matching value
        while bfs:
            idx = bfs.pop(0)
    
            # return if cell of correct value is found
            if flatMap[idx] == val:
                result = idx
                return True, result
    
            # iterate over all adjacent unvisited cells
            for nbr in self.nhood8(idx):
                if not visited_flag[nbr]:
                    bfs.append(nbr)
                    visited_flag[nbr] = True
    
        return False, None
    
    def __str__(self):
        return str(self.map)


"""
Helper Functions
"""
# Euclidian Distance 
def distanceBetweenCoords(a: Point, b: Point) -> float:
    return math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)