from collections import deque
import multiprocessing
import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import math
import os,sys

import tqdm

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

from utils.logger import logger


class BaseSensor:
    def __init__(self) -> None:
        pass

    def retrieve(self):
        raise NotImplementedError()

class State(BaseSensor):
    def __init__(self, client, drone_name=''):
        self.data = {'position': None, 'linear_velocity': None, 'linear_acceleration':None,
                     'orientation':None, 'angular_velocity':None, 'angular_acceleration':None}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name

    def retrieve(self):
        data = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = {}
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision['has_collided'] = collision_info.has_collided
        collision['object_name'] = data.collision.object_name
        gps_location = [data.gps_location.latitude,data.gps_location.longitude,data.gps_location.altitude]
        timestamp = data.timestamp
        position = list(data.kinematics_estimated.position)
        linear_velocity = list(data.kinematics_estimated.linear_velocity)
        linear_acceleration = list(data.kinematics_estimated.linear_acceleration)
        orientation = list(data.kinematics_estimated.orientation)
        angular_velocity = list(data.kinematics_estimated.angular_velocity)
        angular_acceleration = list(data.kinematics_estimated.angular_acceleration)

        self.data.update({'collision': collision, 
                          'gps_location': gps_location,
                          'timestamp': timestamp, 
                          'position': position,
                          'linear_velocity': linear_velocity,
                          'linear_acceleration': linear_acceleration,
                          'orientation': orientation,
                          'angular_velocity': angular_velocity,
                          'angular_acceleration': angular_acceleration
                          })
        return self.data
        
        
class Imu(BaseSensor):
    def __init__(self, client, drone_name='', imu_name=''):
        self.data = {}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name
        self.imu_name = imu_name

    def retrieve(self):
        data = self.client.getImuData(imu_name=self.imu_name,vehicle_name=self.drone_name)
        time_stamp = data.time_stamp
        orientation = data.orientation
        angular_velocity = list(data.angular_velocity)
        linear_acceleration = list(data.linear_acceleration)
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                      [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                      [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)])).tolist()
        self.data.update({'time_stamp': time_stamp, 'rotation': rotation_matrix, 'orientation': list(data.orientation),
                          'linear_acceleration': linear_acceleration, 'angular_velocity': angular_velocity})
        return self.data


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info ]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    while not confirmed and count < 30:
                        try:
                            self.airsim_clients[index_1][index_2].confirmConnection()
                            self.airsim_clients[index_1][index_2].enableApiControl(True)
                            self.airsim_clients[index_1][index_2].armDisarm(True)
                            self.airsim_clients[index_1][index_2].takeoffAsync().join()
                            confirmed = True
                        except Exception as e:
                            time.sleep(1)
                            print('failed', e)
                            count += 1
                            pass
        
        return confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=300) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            logger.info(f'gpus: {self.machines_info[index]}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            print(list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            if result[0] == False:
                logger.error(f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('打开场景失败')
            
            print("===")
            assert len(result[1]) == 2, '打开场景失败'
            print('waiting for airsim connection...')
            time.sleep(3 * len(self.machines_info[index]['open_scenes']) + 10)
            ip = result[1][0]
            ports = result[1][1]
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            self.airsim_ip = ip
            self.airsim_ports = ports
            assert str(ip) == str(socket_client.address._host), '打开场景失败'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:  
                    self.airsim_clients[index][i] = airsim.MultirotorClient(ip=ip, port=port, timeout_value=airsim_timeout)
                    print(port)

            logger.info(f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        after = time.time()
        diff = after - before
        logger.info(f"启动时间：{diff}")

        assert self._confirmConnection(), 'server connect failed'
        self._closeSocketConnection()
    
    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)



    def move_to_next_pose(self, poses_list: list, fly_types: list):
        def _move(airsim_client: airsim.VehicleClient, pose: airsim.Pose, fly_type: str):
            if airsim_client is None:
                raise Exception('error')
                return
            
            state_sensor = State(airsim_client)
            imu_sensor = Imu(airsim_client,imu_name="Imu")
            
            # airsim_client.armDisarm(True)
            if fly_type == 'move':
                drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
                yaw_mode=airsim.YawMode(is_rate=False)
                airsim_client.moveToPositionAsync(pose.position.x_val, pose.position.y_val, pose.position.z_val,
                                                  velocity=1, drivetrain=drivetrain, yaw_mode=yaw_mode)

            elif fly_type == 'rotate':
                (pitch, roll, yaw) = airsim.to_eularian_angles(pose.orientation)
                airsim_client.rotateToYawAsync(math.degrees(yaw))
            
            state_info = copy.deepcopy(state_sensor.retrieve())
            imu_info = copy.deepcopy(imu_sensor.retrieve())
            position = np.array(state_info['position'])

            results = []
            collision = False
            if state_info['collision']['has_collided']:
                collision = True
            results.append({'sensors':{'state':state_info,'imu':imu_info}})
            return {'states': results,'collision':collision}


        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_move, (self.airsim_clients[index_1][index_2], poses_list[index_1][index_2], fly_types[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result =  threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(
                    result
                )
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move by position failed.')
            return None
        if error_flag:
            return None
        return result_poses_list
    
    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose, ) -> None:
            if airsim_client is None:
                raise Exception('error')
                return
            
            airsim_client.simSetVehiclePose(pose=pose, ignore_collision=True)
            vehicles=airsim_client.listVehicles()
            airsim_client.simSetObjectScale(vehicles[0],airsim.Vector3r(0.5,0.5,0.5))
            
            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                print("index_1,index_2: ", str(index_1),str(index_2))
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses失败')
            return False

        return True
    
    def getImageResponses(self, cameras=['0', '1', '2', '3'], poses=None):
        def _getImages(airsim_client: airsim.VehicleClient):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            while True:
                try:
                    ImageRequest = []
                    for camera_name in cameras:
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=True))
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
                    image_datas = airsim_client.simGetImages(requests=ImageRequest)
                    images, depth_images = [], []
                    for idx, camera_name in enumerate(cameras):
                        rgb_resp = image_datas[2 * idx]
                        image = rgb_resp.image_data_uint8
                        depth_resp = image_datas[2* idx + 1]
                        depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)
                    break
                except Exception as e:
                    time_sleep_cnt += 1
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 10:
                    raise Exception('图片获取失败')
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses

    def getSensorInfo(self, ):
        def get_sensor_info(airsim_client: airsim.VehicleClient, ):
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client)
            state_info = state_sensor.retrieve()
            imu_info = imu_sensor.retrieve()
            return {'sensors': {'state':state_info, 'imu': imu_info}}
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(get_sensor_info, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        results = []
        for index_1, _ in enumerate(threads):
            results.append([])
            for index_2, _ in enumerate(threads[index_1]):
                results[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getSensorInfo failed.')
            return None
        return results 

