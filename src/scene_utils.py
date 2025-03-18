import airsim
import random
import json
import os
import numpy as np
import pathlib as Path

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from utils.logger import logger
from src.common.param import args

file_dir = os.path.dirname(os.path.abspath(__file__))
LOCATION_INFO_PATH=os.path.join(file_dir,"location.json")

class SceneInitializer:

    SimulatorClientTool = AirVLNSimulatorClientTool(args.machines_info)
    SimulatorClientTool.run_call()
    
   
    def __init__(self,
                 SceneInfoPath=LOCATION_INFO_PATH,
                 Scene_Name="StreetandPark"):
        self.SceneINfoPath=SceneInfoPath
        self.Scene_Name=Scene_Name
        self.SceneInfo=None
        self.StartPoint=None
        #print(LOCATION_INFO_PATH)
        self.loadDataset(Scene_Name)
        self.initDrone()
        self.initialObjects()

    def loadDataset(self,Scene_Name:str):
        try:
            with open(LOCATION_INFO_PATH, "r") as f:
                file = json.load(f)
                self.SceneInfo=file[Scene_Name]
                #print("SceneInfo: ", self.SceneInfo)
        except Exception as e:
            print("Occuered error in reading location file:"+str(e))
            logger.error("Occuered error in reading location file:"+str(e))



    def initDrone(self):
        Drones_Location=self.SceneInfo["start_points"]
        StartPoints=list(Drones_Location.values())
        StartPoint=random.choice(StartPoints)
        pose=airsim.Pose(airsim.Vector3r(StartPoint["x_val"], StartPoint["y_val"], StartPoint["z_val"]))
        #print("pose: ", pose)
        self.SimulatorClientTool.setPoses([[pose]])    

    def initialObjects(self):
        Objects_Info=self.SceneInfo["objects"]
        self.SimulatorClientTool.setObjects(Objects_Info)

    def getTargetList(self, targets:list):
        """Gets a list of targets within a certain range.

            Args:
                targets (List[Any]): A list of potential targets.

            Returns:
                List[Any]: A list of targets that are within the specified distance.

        """

    def getInstructionList(self, targets:list):
        """Gets instruction for each target from json.

            Args:
                targets (List[Any]): A list of targets.

            Returns:
                List[Any]: A list of instruction for each target.

        """

    def calculateDistance(self, target:dict):
        """Calculates the distance between the drone and the target.

            Args:
                target (Dict): The target object.

            Returns:
                float: The distance between the drone and the target.

        """
