
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import roma
import torch.nn.functional as F
from loguru import logger
import torch.nn as nn
import torch.optim as optim
import numpy as np
import motionblender.lib.misc as misc
from motionblender.app.utils import RobotInterface, MotionBlender
import motionblender.lib.animate as anim
from jaxtyping import Float32
from tqdm.auto import tqdm, trange
import kinpy as kp
import torch
import transformations as tf
from torch import Tensor

def map_gripper_ctrl_to_joint_v(gv):
    v = max(min(gv, 100), 30)
    v = 0.8 * (v - 30) / 70
    return {
        'finger_1_joint_1': v,
        'finger_2_joint_1': v,
        'finger_middle_joint_1': v
    }

robot_joints = [
                'world_joint', # fixed
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint',
                'ee_fixed_joint', # fixed
                'bracket_to_dh13', # fixed
                'right_index_joint_0',
                'right_index_joint_1',
                'right_index_joint_2',
                'right_index_joint_3',
                'right_middle_joint_0',
                'right_middle_joint_1',
                'right_middle_joint_2',
                'right_middle_joint_3',
                'right_ring_joint_0',
                'right_ring_joint_1',
                'right_ring_joint_2',
                'right_ring_joint_3',
                'right_thumb_joint_0',
                'right_thumb_joint_1',
                'right_thumb_joint_2',
                'right_thumb_joint_3'
                ]


robot_joints_from_origin = {                
            'world_joint': 'base_link', # fixed
            'shoulder_pan_joint': 'shoulder_link',
            'shoulder_lift_joint': 'upper_arm_link',
            'elbow_joint': 'forearm_link',
            'wrist_1_joint': 'wrist_1_link',
            'wrist_2_joint': 'wrist_2_link',
            'wrist_3_joint': 'wrist_3_link',
            'ee_fixed_joint': 'ee_link', # fixed
            'bracket_to_dh13': 'right_palm_link', # fixed
            'right_index_joint_0': 'right_index_link_0',
            'right_index_joint_1': 'right_index_link_1',
            'right_index_joint_2': 'right_index_link_2',
            'right_index_joint_3': 'right_index_link_3',
            'right_middle_joint_0': 'right_middle_link_0',
            'right_middle_joint_1': 'right_middle_link_1',
            'right_middle_joint_2': 'right_middle_link_2',
            'right_middle_joint_3': 'right_middle_link_3',
            'right_ring_joint_0': 'right_ring_link_0',
            'right_ring_joint_1': 'right_ring_link_1',
            'right_ring_joint_2': 'right_ring_link_2',
            'right_ring_joint_3': 'right_ring_link_3',
            'right_thumb_joint_0': 'right_thumb_link_0',
            'right_thumb_joint_1': 'right_thumb_link_1',
            'right_thumb_joint_2': 'right_thumb_link_2',
            'right_thumb_joint_3': 'right_thumb_link_3'
        }

robot_connections = [
    ('world_joint', 'shoulder_pan_joint'),
    ('shoulder_pan_joint', 'shoulder_lift_joint'),
    ('shoulder_lift_joint', 'elbow_joint'),
    ('elbow_joint', 'wrist_1_joint'),
    ('wrist_1_joint', 'wrist_2_joint'),
    ('wrist_2_joint', 'wrist_3_joint'),
    ('wrist_3_joint', 'ee_fixed_joint'),
    ('ee_fixed_joint', 'bracket_to_dh13'),

    ('bracket_to_dh13', 'right_index_joint_0'),
    ('right_index_joint_0', 'right_index_joint_1'),
    ('right_index_joint_1', 'right_index_joint_2'),
    ('right_index_joint_2', 'right_index_joint_3'),

    ('bracket_to_dh13', 'right_middle_joint_0'),
    ('right_middle_joint_0', 'right_middle_joint_1'),
    ('right_middle_joint_1', 'right_middle_joint_2'),
    ('right_middle_joint_2', 'right_middle_joint_3'),

    ('bracket_to_dh13', 'right_ring_joint_0'),
    ('right_ring_joint_0', 'right_ring_joint_1'),
    ('right_ring_joint_1', 'right_ring_joint_2'),
    ('right_ring_joint_2', 'right_ring_joint_3'),

    ('bracket_to_dh13', 'right_thumb_joint_0'),
    ('right_thumb_joint_0', 'right_thumb_joint_1'),
    ('right_thumb_joint_1', 'right_thumb_joint_2'),
    ('right_thumb_joint_2', 'right_thumb_joint_3')
]

robot_connections_int = [(robot_joints.index(a), robot_joints.index(b)) for a, b in robot_connections]

default_joint_values = {

    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': 0.0,
    'elbow_joint': 0.0,
    'wrist_1_joint': 0.0,
    'wrist_2_joint': 0.0,
    'wrist_3_joint': 0.0,

    'right_index_joint_0': 0.0,
    'right_index_joint_1': 0.17453292519943295,
    'right_index_joint_2': 0.17453292519943295,
    'right_index_joint_3': 0.0,
    'right_middle_joint_0': 0.0,
    'right_middle_joint_1': 0.17453292519943295,
    'right_middle_joint_2': 0.17453292519943295,
    'right_middle_joint_3': 0.0,
    'right_ring_joint_0': 0.0,
    'right_ring_joint_1': 0.17453292519943295,
    'right_ring_joint_2': 0.17453292519943295,
    'right_ring_joint_3': 0.0,
    'right_thumb_joint_0': 0.0,
    'right_thumb_joint_1': 0.0,
    'right_thumb_joint_2': 0.0,
    'right_thumb_joint_3': 0.0
}

class UR5(RobotInterface):
    def __init__(self, ur5_path="my_data/ur5_dh13_combined.urdf"):
        super().__init__()
        txt = open(ur5_path, 'rb').read()
        self.ur5_serial_chain = kp.build_serial_chain_from_urdf(txt, 'right_palm_link')
        self.ur5_chain = kp.build_chain_from_urdf(txt)
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.curr_jvs = np.array([default_joint_values[jname] for jname in self.joint_names])
        self.buf['rot6d'] = None
    
    def _joint_values_to_rot6d(self, joint_values):
        link_poses = self.ur5_chain.forward_kinematics(joint_values)
        robot_joints_values = {}
        for rj_name, parents in robot_joints_from_origin.items(): 
            if isinstance(parents, list):
                values = []
                for p in parents:
                    values.append(link_poses[p].pos)
                robot_joints_values[rj_name] = sum(values) / len(values)
            else:
                robot_joints_values[rj_name] = link_poses[parents].pos

        joint_positions = []
        for rj_name in robot_joints:
            joint_positions.append(robot_joints_values[rj_name])
        joint_positions = torch.from_numpy(np.array(joint_positions)).float()
        anim_chain = anim.inverse_kinematic(joint_positions, robot_connections_int)
        rot6d = anim.retrieve_tensor_from_chain(anim_chain, 'rot6d')
        return rot6d

    def initialize(self, motion_module: MotionBlender) -> None:
        gripper_poses = getattr(motion_module, 'gripper_poses', [])
        if len(gripper_poses) == 0:
            if hasattr(motion_module, 'original_cano_t'):
                original_cano_t = motion_module.original_cano_t
            else:
                original_cano_t = 323

            ROBOT_ROOT_DATASET_DIR = './datasets/robot/'
            img_id = misc.load_json(osp.join(ROBOT_ROOT_DATASET_DIR, 'dataset.json'))['ids'][original_cano_t]
            jv = misc.load_cpkl(osp.join(ROBOT_ROOT_DATASET_DIR, 'robot_rawdata.pkl'))['joint_pos_list'][int(img_id)]
            self.curr_jvs = np.array(jv)
            end_pose_tf = self.ur5_serial_chain.forward_kinematics(self.curr_jvs)
            ee_pose_robot_base = torch.from_numpy(end_pose_tf.matrix()).float().to(next(motion_module.parameters()).device)
            self.buf['pose'] = ee_pose_robot_base
            self.buf['degree'] = 50.
        else:
            self.buf['pose'] = gripper_poses[0]
            self.buf['degree'] = motion_module.gripper_degrees[0]

        self.get_joint_rotations()
        self.inited = True
    
    def get_joint_rotations(self, refresh=False) -> Float32[Tensor, "j 6"]:
        if not self.buf['stale'] and self.buf['rot6d'] is not None and not refresh:
            return self.buf['rot6d']

        matrix = self.buf['pose'].cpu().numpy()
        trans = kp.Transform(rot=tf.quaternion_from_matrix(matrix), pos=matrix[:3, 3])
        self.curr_jvs = self.ur5_serial_chain.inverse_kinematics(trans, self.curr_jvs)
        joint_values = {**default_joint_values, **map_gripper_ctrl_to_joint_v(self.buf['degree']), 
                        **dict(zip(self.joint_names, self.curr_jvs))}
        rot6d = self._joint_values_to_rot6d(joint_values)
        self.buf['rot6d'] = rot6d.to(self.buf['pose'].device)
        self.buf['stale'] = False
        return self.buf['rot6d']

robot = UR5()

# if __name__ == "__main__":
#     # gs_modules, motion_modules, _, gaussian_names = misc.load_cpkl("outputs/mb/robot/okish/toy/ckpt.robot.cpkl")
#     # robot.initialize(motion_modules['robot'])
#     robot.buf['pose'][:3, 3] += (torch.rand(3).cuda() * 0.3)
#     robot.buf['pose'][:3, :3] = roma.random_rotmat().cuda() @ robot.buf['pose'][:3, :3]
#     print(robot.get_joint_rotations(refresh=True))
