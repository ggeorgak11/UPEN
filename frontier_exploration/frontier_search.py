"""
Implementation of a frontier-search task for an input semantic map

Directly Adapted from: http://wiki.ros.org/frontier_exploration (C++ and costmap_2d)
Based on: Brian Yamauchi, FBE (http://www.robotfrontier.com/papers/cira97.pdf)
"""

from frontier_exploration.map import *
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors

VOID = 0
OCCUPIED = 1
FREE = 2


class FrontierSearch():
    # Public Functions
    """
    * @brief Constructor for search task
    * @param costmap Reference to costmap data to search.
    * @param min_frontier_size The minimum size to accept a frontier
    * @param travel_point The requested travel point (closest|middle|centroid)
    """
    def __init__(self, episode_idx, step, step_ego_grid_crops_3: np.ndarray, min_frontier_size: int, travel_point: str):
        self.idx = episode_idx
        self.step = step
        self.map = Map(step_ego_grid_crops_3)
        self.flatMap = self.map.getCharMap()
        self.size_x, self.size_y = self.map.getSizeInCells()
        self.min_frontier_size = min_frontier_size
        self.travel_point = travel_point
        self.frontier_arr = None
        self.random_magnitude = 15
        
    """
    next long term goal
    """
    def nextGoal(self, pose_coords, _rel_pose, min_thresh: int = 4):
        frontiers = self.searchFrom(pose_coords)
        if len(frontiers) == 0:
            ori = _rel_pose[0,2] + math.pi / 2
            x = math.cos(math.pi *5/4)
            y = math.sin(math.pi *5/4)
            opposite_dir = [[[ -x * self.random_magnitude, -y * self.random_magnitude]]]
            next_goal = pose_coords + opposite_dir
            print("going backward")
            return next_goal
        else:
            closest_frontier = None 
            # pick a frontier that is AT LEAST min_thresh cells away 
            # (so it's not just the frontier that's right infront of the agent's unexplored circle)
            for frontier in frontiers:
                if frontier.min_distance >= min_thresh:
                    closest_frontier = frontier
                    break                    
            if closest_frontier is None:
                print("no frontier within thresh. picking furthest")
                closest_frontier = frontiers[-1]
            return np.array([[[closest_frontier.travel_point.x, closest_frontier.travel_point.y]]])
            
    """
    * @brief Runs search implementation, outward from the start position
    * @param position Initial position to search from
    * @return List of frontiers, if any
    """
    def searchFrom(self, pose_coords) -> list:
        frontier_list = []
        mx, my = pose_coords[0][0][0], pose_coords[0][0][1]    
        
        # initialize flag arrays to keep track of visited and frontier cells
        frontier_flag = [False] * (self.size_x * self.size_y)
        visited_flag = [False] * (self.size_x * self.size_y)
    
        # initialize breadth first search
        bfs = []
        
        # find nearest_clear cell to start search
        agent_pos = self.map.getIndex(mx, my)
        found, nearest_clear = self.map.nearestCell(agent_pos, FREE)
        if found:
            bfs.append(nearest_clear)
        else:
            bfs.append(agent_pos)
            print("Could not find nearby clear cell to start search")
    
        visited_flag[bfs[0]] = True

        while bfs:
            idx = bfs.pop(0)

            for nbr in self.map.nhood8(idx): 
                # add to queue all free, unvisited cells, use descending search in case initialized on non-free cell
#                if self.flatMap[nbr] <= self.flatMap[idx] and not visited_flag[nbr]:
                if self.flatMap[nbr] == FREE and not visited_flag[nbr]:
                    visited_flag[nbr] = True
                    bfs.append(nbr)
                # check if cell is new frontier cell (unvisited, VOID, free neighbour)
                if self.isNewFrontierCell(nbr, frontier_flag):
                    frontier_flag[nbr] = True
                    new_frontier = self.buildNewFrontier(nbr, agent_pos, frontier_flag)
                    if new_frontier.size > self.min_frontier_size:
                        frontier_list.append(new_frontier)
        self.frontier_arr = np.asarray(frontier_flag).reshape((self.size_y, self.size_x))
        return frontier_list 
    

    # Protected Functions 
    """
    * @brief Starting from an initial cell, build a frontier from valid adjacent cells
    * @param initial_cell Index of cell to start frontier building
    * @param reference Reference index to calculate position from
    * @param frontier_flag Flag vector indicating which cells are already marked as frontiers
    * @return
    """
    def buildNewFrontier(self, initial_cell: int, reference: int, frontier_flag: list) -> Frontier:
        # initialize frontier structure
        output = Frontier()
    
        centroid, middle = Point(), Point()

        # record initial contact point for frontier
        initial_point = self.map.indexToPoint(initial_cell)
        output.travel_point = initial_point.copy()
        output.points.append(initial_point)
        # push initial gridcell onto queue
        bfs = []
        bfs.append(initial_cell)

        # cache reference position in world coords
        agent_point = self.map.indexToPoint(reference)
        output.min_distance = distanceBetweenCoords(initial_point, agent_point)
        
        while bfs:
            idx = bfs.pop(0)

            # try adding cells in 8-connected neighborhood to frontier
            for nbr in self.map.nhood8(idx):
                # check if neighbour is a potential frontier cell
                if self.isNewFrontierCell(nbr, frontier_flag):
                    # mark cell as frontier
                    frontier_flag[nbr] = True
                    w = self.map.indexToPoint(nbr)

                    # update frontier size
                    output.size += 1
                    output.points.append(self.map.indexToPoint(nbr) )

                    # determine frontier's distance from robot, going by closest gridcell to robot
                    distance = distanceBetweenCoords(w, agent_point)
                    if distance < output.min_distance:
                        output.min_distance = distance
                        middle.x = w.x
                        middle.y = w.y

                    # add to queue for breadth first search
                    bfs.append(nbr)
    
        # average out frontier centroid
        if (self.travel_point == "closest"):
            pass
            # point already set
        elif (self.travel_point == "middle"):
            output.travel_point = middle
        elif (self.travel_point == "centroid"):
            # find centroid of output.points
            for point in output.points:
                centroid.x += point.x
                centroid.y += point.y
            centroid.x /= output.size
            centroid.y /= output.size                
            output.travel_point = centroid
        else:
            print("Invalid 'frontier_travel_point' parameter, falling back to 'closest'");
            # point already set
    
        return output
    
        
    """
    * @brief isNewFrontierCell Evaluate if candidate cell is a valid candidate for a new frontier.
    * @param idx Index of candidate cell
    * @param frontier_flag Flag vector indicating which cells are already marked as frontiers
    * @return
    """
    def isNewFrontierCell(self, idx: int, frontier_flag: list) -> bool:
        # check that cell is unknown and not already marked as frontier
        if self.flatMap[idx] != VOID or frontier_flag[idx]:
            return False
    
        for nbr in self.map.nhood4(idx):
            if self.flatMap[nbr] == FREE:
                return True
        return False
    
    
    def save_frontier_labeled_map(self, save_img_dir_, t, pose_coords, ltg):
        cmap = colors.ListedColormap(['grey', 'red', 'green', 'yellow'])
        bounds = [-.1,.4,1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots()
        ax.imshow(self.map.map, cmap=cmap, norm=norm)
        height, width = self.map.map.shape
        ax.axis('off')

        plt.scatter(ltg[0,0,0], ltg[0,0,1], color="magenta", s=1)
        plt.scatter(pose_coords[0,0,0], pose_coords[0,0,1], color="blue", s=1)
    
        plt.savefig(save_img_dir_ + 'fbe_' + str(t) + 'labels.png', bbox_inches='tight', pad_inches=0, dpi=200)

        
        print("generated: ", save_img_dir_ + 'fbe_' + str(t))
        # FRONTIERS
        if self.frontier_arr is not None:
            fig, ax = plt.subplots()
            ax.imshow(self.map.map + self.frontier_arr * 3, cmap=cmap, norm=norm)
            ax.axis('off')

            plt.scatter(ltg[0,0,0], ltg[0,0,1], color="magenta", s=1)
            plt.scatter(pose_coords[0,0,0], pose_coords[0,0,1], color="blue", s=1)
        
            plt.savefig(save_img_dir_ + 'fbe_' + str(t) + 'frontiers.png', bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close()