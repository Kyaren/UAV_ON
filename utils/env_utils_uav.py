import math
import numba as nb
import airsim
import numpy as np
import copy

from src.common.param import args
from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings
from utils.logger import logger


class SimState:
    def __init__(self, index=-1,                                                                                                                                                        
                 step=0,
                  
                 ):
        self.index = index
        self.step = step
        
        self.is_end = False
        self.oracle_success = False
        self.is_collisioned = False
        self.predict_start_index = 0
        self.history_start_indexes = [0]
        self.SUCCESS_DISTANCE = 20
        self.progress = 0.0
        self.waypoint = {}
        self.sensorInfo = {}
    
    
    @property
    def state(self): 
        return self.trajectory[-1]['sensors']['state']

    @property
    def pose(self): # 
        return self.trajectory[-1]['sensors']['state']['position'] + self.trajectory[-1]['sensors']['state']['orientation']

class ENV:
    def __init__(self, load_scenes: list):
        self.batch = None

    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)
        return

    def get_obs_at(self, index: int, state: SimState):
        assert self.batch is not None, 'batch is None'
        item = self.batch[index]
        oracle_success = state.oracle_success
        done = state.is_end
        return (done, oracle_success), state

def getNextPosition(current_pose: airsim.Pose, action, step_size):
    current_position = np.array([current_pose.position.x_val, current_pose.position.y_val, current_pose.position.z_val])
    current_orientation =current_pose.orientation #order is w,x,y,z

    (pitch, roll, yaw) = airsim.to_eularian_angles(current_orientation)

    if action == AirsimActions.MOVE_FORWARD:
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)

        vector = np.array([dx, dy, dz])
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            unit_vector = vector / norm
        else:
            unit_vector = np.array([0, 0, 0])
        
        new_position = current_position + unit_vector * step_size
        new_orientation = current_orientation
        fly_type = "move"

    elif action == AirsimActions.TURN_LEFT:
        new_yaw = yaw - math.radians(step_size)
        if math.degrees(new_yaw) < -180:
            new_yaw += math.radians(360)
        
        new_position = current_position
        new_orientation = airsim.to_quaternion(pitch, roll, new_yaw)
        fly_type = "rotate"

    elif action == AirsimActions.TURN_RIGHT:
        new_yaw = yaw + math.radians(step_size)
        if math.degrees(new_yaw) > 180:
            new_yaw -= math.radians(360)
        
        new_position = current_position
        new_orientation = airsim.to_quaternion(pitch, roll, new_yaw)
        fly_type = "rotate"

    elif action == AirsimActions.GO_UP:
        unit_vector = np.array([0, 0, -1])
        
        new_position = current_position + unit_vector * step_size
        new_orientation = current_orientation
        fly_type = "move"

    elif action == AirsimActions.GO_DOWN:
        unit_vector = np.array([0, 0, 1])

        new_position = current_position + unit_vector * step_size
        new_orientation = current_orientation
        fly_type = "move"

    #elif action == AirsimActions.MOVE_LEFT:
        
    new_pose = airsim.Pose(
        airsim.Vector3r(new_position[0], new_position[1], new_position[2]),
        airsim.Quaternionr(new_orientation.x_val, new_orientation.y_val, new_orientation.z_val, new_orientation.w_val)
    )
    return (new_pose, fly_type)