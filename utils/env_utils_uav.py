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
                 task_info ={}
                 ):
        self.index = index
        self.step = step
        self.task_info = task_info
        self.is_end = False
        self.oracle_success = False
        self.is_collisioned = False
        self.predict_start_index = 0
        self.history_start_indexes = [0]
        self.SUCCESS_DISTANCE = 20
        self.progress = 0.0
        self.waypoint = {}
        self.sensorInfo = {}
        self.start_pose = task_info['start_pose']
        self.trajectory = [{'sensors': {'state': {'position': self.start_pose['start_position'], 'quaternionr': self.start_pose['start_quaternionr']}}}]
        self.move_distance = 0.0
        self.heading_changes :list[float] = []
    
    @property
    def state(self): 
        return self.trajectory[-1]['sensors']['state']

    @property
    def pose(self): # 
        return self.trajectory[-1]['sensors']['state']['position'] + self.trajectory[-1]['sensors']['state']['quaternionr']
    

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

def getNextPosition(current_pose: airsim.Pose, action, step_size, is_fixed):
    current_position = np.array([current_pose.position.x_val, current_pose.position.y_val, current_pose.position.z_val])
    current_orientation =current_pose.orientation #order is w,x,y,z

    (pitch, roll, yaw) = airsim.to_eularian_angles(current_orientation)

    if action == 'forward':
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)

        vector = np.array([dx, dy, dz])
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            unit_vector = vector / norm
        else:
            unit_vector = np.array([0, 0, 0])
        
        if is_fixed:
            new_position = current_position + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
        else:
            new_position = current_position + unit_vector * step_size
        
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'backward':
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)

        vector = np.array([dx, dy, dz])
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            unit_vector = vector / norm
        else:
            unit_vector = np.array([0, 0, 0])
        
        if is_fixed:
            new_position = current_position - unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
        else:
            new_position = current_position - unit_vector * step_size
        
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'rotl':
        if is_fixed:
            new_yaw = yaw - math.radians(AirsimActionSettings.TURN_ANGLE)
        else:
            new_yaw = yaw - math.radians(step_size)
        
        if math.degrees(new_yaw) < -180:
            new_yaw += math.radians(360)
        
        new_position = current_position
        new_orientation = airsim.to_quaternion(pitch, roll, new_yaw)
        fly_type = "rotate"

    elif action == 'rotr':
        if is_fixed:
            new_yaw = yaw + math.radians(AirsimActionSettings.TURN_ANGLE)
        else:    
            new_yaw = yaw + math.radians(step_size)
        
        if math.degrees(new_yaw) > 180:
            new_yaw -= math.radians(360)
        
        new_position = current_position
        new_orientation = airsim.to_quaternion(pitch, roll, new_yaw)
        fly_type = "rotate"

    elif action == 'ascend':
        unit_vector = np.array([0, 0, -1])
        
        if is_fixed:
            new_position = current_position + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE
        else:
            new_position = current_position + unit_vector * step_size
        
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'descend':
        unit_vector = np.array([0, 0, 1])

        if is_fixed:
            new_position = current_position + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE
        else:    
            new_position = current_position + unit_vector * step_size
        
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'left':
        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        vector = np.array([unit_x, unit_y, 0])

        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            unit_vector = vector / norm
        else:
            unit_vector = np.array([0, 0, 0])
        
        if is_fixed:
            new_position = current_position - unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
        else:
            new_position = current_position - unit_vector * step_size

        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'right':
        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        vector = np.array([unit_x, unit_y, 0])

        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            unit_vector = vector / norm
        else:
            unit_vector = np.array([0, 0, 0])
        
        if is_fixed:
            new_position = current_position + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
        else:
            new_position = current_position + unit_vector * step_size

        new_orientation = current_orientation
        fly_type = "move"
    else:
        new_position = current_position
        new_orientation = current_orientation
        fly_type = "stop"

    new_pose = airsim.Pose(
        airsim.Vector3r(new_position[0], new_position[1], new_position[2]),
        airsim.Quaternionr(new_orientation.x_val, new_orientation.y_val, new_orientation.z_val, new_orientation.w_val)
    )
    return (new_pose, fly_type)