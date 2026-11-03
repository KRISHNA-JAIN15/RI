"""DDPG path planning for a six-axis robotic arm in a multi-obstacle environment.

Re-implementation of Li et al., "Deep Reinforcement Learning for Robotic Arm Path
Planning in Multi-Obstacle Environments", ACIRS 2024.
"""

from arm_rl.geometry import closest_points_between_segments, segment_distance_matrix
from arm_rl.kinematics import SixAxisArm
from arm_rl.scene import FruitTreeSceneGenerator, Scene
from arm_rl.env import ArmPathPlanningEnv
from arm_rl.networks import Actor, Critic
from arm_rl.replay_buffer import ReplayBuffer
from arm_rl.ddpg import DDPGAgent
from arm_rl.rrt import RRTPlanner

__all__ = [
    "closest_points_between_segments",
    "segment_distance_matrix",
    "SixAxisArm",
    "FruitTreeSceneGenerator",
    "Scene",
    "ArmPathPlanningEnv",
    "Actor",
    "Critic",
    "ReplayBuffer",
    "DDPGAgent",
    "RRTPlanner",
]
