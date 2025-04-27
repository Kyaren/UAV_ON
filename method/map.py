import numpy as np


class gridMap:
    def __init__(self, vehcile_height:float, 
                 feature_dim:int, origin,
                 search_diameter:int):
        """
            args:
                map_width[int]: number of width direction grids
                vehcile_height[int]: height of vehicle
                feature_dim[int]: dimension of features for each grid 
                
                orgin[list]: position in simulation environment and the (0,0) position in grid map
                search_diameter[int]: diameter of search area
        """
        self.vehcile_height=vehcile_height
        self.feature_dim=feature_dim
        self.origin=origin
        self.search_diameter=search_diameter
        self.map_width=(search_diameter+(2*vehcile_height)-1)//(2*vehcile_height)
        #self.gridMap=np.zeros((map_width, map_width, feature_dim), dtype=np)

    def initializeMap(self):
        if(self.map_width%2==0):
            self.map_width+=1

        self.gridMap=np.zeros((self.map_width, self.map_width, self.feature_dim), dtype=[
            ('explored', bool),
            ('obstacle', bool),
            ('observation', np.float32)
        ])

    def getCenterIndex(self):
        index = self.map_width // 2 + 1
        return index, index


    def worldToGrid(self, position:list):
        """
            Convert position in simulation environment to position in grid map

            args:
                position[list]: position in simulation environment
            return:
                grid_position[list]: position in grid map
        """
        
    def updateMap(self, observation):
        """
            Update grid map with new observation

            args:
                observation[ ]: observation from sensors

            return: None
        """


    def isValid(self, position:list):
        """
            Check if the position is valid in grid map (检查是否越界)

            args:
                position[list]: position in grid map

            return:
                bool: True if the position is in the grid map, False otherwise
        """

    def getNearbyGridsInfo(self, index: tuple[int, int]):
        """
            Get information of nearby grids, (考虑是否在地图边界)

            args:
                index[tuple]: index of grid

            return:
                nearby_grids_info[list]: information of nearby grids
        """

    