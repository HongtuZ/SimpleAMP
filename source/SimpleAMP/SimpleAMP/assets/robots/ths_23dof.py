import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from SimpleAMP import ROOT_DIR

ARMATURE_E00 = 0.001 * 2
ARMATURE_E02 = 0.0042
ARMATURE_E03 = 0.02
ARMATURE_E06 = 0.012

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz     = 62.83185307    平方 = 3947.8417602
DAMPING_RATIO = 2.0

STIFFNESS_E00 = ARMATURE_E00 * NATURAL_FREQ**2  # 3.94784176       # 4
STIFFNESS_E02 = ARMATURE_E02 * NATURAL_FREQ**2  # 16.58093539      # 16.6
STIFFNESS_E03 = ARMATURE_E03 * NATURAL_FREQ**2  # 78.956835204     # 79
STIFFNESS_E06 = ARMATURE_E06 * NATURAL_FREQ**2  # 47.3741011224    #

DAMPING_E00 = 2.0 * DAMPING_RATIO * ARMATURE_E00 * NATURAL_FREQ  # 0.2513274122
DAMPING_E02 = 2.0 * DAMPING_RATIO * ARMATURE_E02 * NATURAL_FREQ  # 1.0555751315
DAMPING_E03 = 2.0 * DAMPING_RATIO * ARMATURE_E03 * NATURAL_FREQ  # 5.0265482456
DAMPING_E06 = 2.0 * DAMPING_RATIO * ARMATURE_E06 * NATURAL_FREQ  # 3.01592894736

THS23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROOT_DIR / "robot_assets/ths_23dof/usd/ths_23dof.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),  # (0.0, 0.0, 0.752),
        joint_pos={
            "left_hip_pitch_joint": 0.25,  # 03
            "left_hip_roll_joint": 0.0,  # 03
            "left_hip_yaw_joint": 0.0,  # 03
            "left_knee_joint": -0.6,  # 03
            "left_ankle_pitch_joint": -0.35,  # 02
            "left_ankle_roll_joint": 0.0,  # 00
            "right_hip_pitch_joint": -0.25,  # 03
            "right_hip_roll_joint": 0.0,  # 03
            "right_hip_yaw_joint": 0.0,  # 03
            "right_knee_joint": 0.6,  # 03
            "right_ankle_pitch_joint": 0.35,  # 02
            "right_ankle_roll_joint": 0.0,  # 00
            "torso_joint": 0.0,  # 06
            "left_shoulder_pitch_joint": 0.0,  # 02
            "left_shoulder_roll_joint": 1.4,  # 00
            "left_shoulder_yaw_joint": 0.0,  # 00
            "left_elbow_joint": 0.3,  # 00
            "left_wrist_roll_joint": 0.0,  # 00
            "right_shoulder_pitch_joint": 0.0,  # 02
            "right_shoulder_roll_joint": -1.4,  # 00
            "right_shoulder_yaw_joint": 0.0,  # 00
            "right_elbow_joint": -0.3,  # 00
            "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            # 使用正则表达式匹配腿部相关关节名称
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            # 仿真中的力矩限制
            effort_limit_sim={
                ".*_hip_pitch_joint": 50,
                ".*_hip_roll_joint": 50,
                ".*_hip_yaw_joint": 50,
                ".*_knee_joint": 50,
            },
            # 仿真中的最大速度限制
            velocity_limit_sim={
                ".*_hip_pitch_joint": 10,
                ".*_hip_roll_joint": 10,
                ".*_hip_yaw_joint": 10,
                ".*_knee_joint": 10,
            },
            # 关节刚度系数
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_E03,  # 100.0,
                ".*_hip_roll_joint": STIFFNESS_E03,  # 100.0,
                ".*_hip_yaw_joint": STIFFNESS_E03,  # 100.0,
                ".*_knee_joint": STIFFNESS_E03,  # 150.0,
            },
            # 关节阻尼系数
            damping={
                ".*_hip_pitch_joint": DAMPING_E03,  # 3.5,
                ".*_hip_roll_joint": DAMPING_E03,  # 3.5,
                ".*_hip_yaw_joint": DAMPING_E03,  # 3.5,
                ".*_knee_joint": DAMPING_E03,  # 5.0,
            },
            # 电气系数（模拟电机内部的反作用力，增加系统的稳定性）
            armature=ARMATURE_E03,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 15,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 10,
            },
            stiffness={
                ".*_ankle_pitch_joint": STIFFNESS_E02,  # 30,
            },
            damping={
                ".*_ankle_pitch_joint": DAMPING_E02,  # 1.5,
            },
            armature=ARMATURE_E02,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet_roll": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_roll_joint": 12,
            },
            velocity_limit_sim={
                ".*_ankle_roll_joint": 10,
            },
            stiffness={
                ".*_ankle_roll_joint": STIFFNESS_E00,  # 20,
            },
            damping={
                ".*_ankle_roll_joint": DAMPING_E00,  # 1.0,
            },
            armature=ARMATURE_E00,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
        "torso": DelayedPDActuatorCfg(
            joint_names_expr=[
                "torso_joint",
            ],
            effort_limit_sim={
                "torso_joint": 20,
            },
            velocity_limit_sim={
                "torso_joint": 10,
            },
            stiffness={
                "torso_joint": STIFFNESS_E06,  # 40,
            },
            damping={
                "torso_joint": DAMPING_E06,  # 2.0,
            },
            armature=ARMATURE_E06,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 15,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 10,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_E02,  # 30,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_E02,  # 1.5,
            },
            armature=ARMATURE_E02,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms_other": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_roll_joint": 12,
                ".*_shoulder_yaw_joint": 12,
                ".*_elbow_joint": 12,
            },
            velocity_limit_sim={
                ".*_shoulder_roll_joint": 10,
                ".*_shoulder_yaw_joint": 10,
                ".*_elbow_joint": 10,
            },
            stiffness={
                ".*_shoulder_roll_joint": STIFFNESS_E00,  # 30,
                ".*_shoulder_yaw_joint": STIFFNESS_E00,  # 15,
                ".*_elbow_joint": STIFFNESS_E00,  # 15,
            },
            damping={
                ".*_shoulder_roll_joint": DAMPING_E00,  # 1.5,
                ".*_shoulder_yaw_joint": DAMPING_E00,  # 1.0,
                ".*_elbow_joint": DAMPING_E00,  # 1.0,
            },
            armature=ARMATURE_E00,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms_end": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": 10,
            },
            stiffness={
                ".*_wrist_roll_joint": STIFFNESS_E00,  # 15,
            },
            damping={
                ".*_wrist_roll_joint": DAMPING_E00,  # 1.0,
            },
            armature=ARMATURE_E00,  # 0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


"""
#----------------------------lab or mujoco/urdf 顺序对照表----------------------------
#  isaac
0'left_hip_pitch_joint', 
1'right_hip_pitch_joint', 
2'torso_joint', 
3'left_hip_roll_joint', 
4'right_hip_roll_joint', 
5'left_shoulder_pitch_joint', 
6'right_shoulder_pitch_joint', 
7'left_hip_yaw_joint', 
8'right_hip_yaw_joint', 
9'left_shoulder_roll_joint', 
10'right_shoulder_roll_joint', 
11'left_knee_joint', 
12'right_knee_joint', 
13'left_shoulder_yaw_joint', 
14'right_shoulder_yaw_joint', 
15'left_ankle_pitch_joint', 
16'right_ankle_pitch_joint', 
17'left_elbow_joint', 
18'right_elbow_joint', 
19'left_ankle_roll_joint', 
20'right_ankle_roll_joint', 
21'left_wrist_roll_joint', 
22'right_wrist_roll_joint'


#  mujoco_idx 
0"left_hip_pitch_joint": 0.3,                           # 左髋俯仰关节初始角度                        
1"left_hip_roll_joint": 0.0,                            # 左髋横滚关节初始角度
2"left_hip_yaw_joint": 0.0,                             # 左髋偏航关节初始角度
3"left_knee_joint": -0.6,                               # 左膝俯仰关节初始角度 
4"left_ankle_pitch_joint": -0.3,                        # 左脚踝俯仰关节初始角度 
5"left_ankle_roll_joint": 0.0,                          # 左脚踝横滚关节初始角度 
6"right_hip_pitch_joint": -0.3,                         # 右髋俯仰关节初始角度 
7"right_hip_roll_joint": 0.0,                           # 右髋横滚关节初始角度
8"right_hip_yaw_joint": 0.0,                            # 右髋偏航关节初始角度 
9"right_knee_joint": 0.6,                               # 右膝俯仰关节初始角度 
10"right_ankle_pitch_joint": 0.3,                        # 右脚踝俯仰关节初始角度 
11"right_ankle_roll_joint": 0.0,                         # 右脚踝横滚关节初始角度 
12"torso_joint": 0.0,                                     # 躯干(腰部)
13"left_shoulder_pitch_joint": 0.0,                      # 左肩俯仰关节初始角度 
14"left_shoulder_roll_joint": 1.3,                       # 左肩横滚关节初始角度 
15"left_shoulder_yaw_joint": -0.6,                       # 左肩偏航关节初始角度 
16"left_elbow_joint": 0.3,                               # 左肘俯仰关节初始角度 
17"left_wrist_roll_joint": 0.0,                           # 左手腕横滚
18"right_shoulder_pitch_joint": 0.0,                      # 右肩俯仰关节初始角度 
19"right_shoulder_roll_joint": -1.3,                      # 右肩横滚关节初始角度 
20"right_shoulder_yaw_joint": 0.6,                        # 右肩偏航关节初始角度 
21"right_elbow_joint": -0.3,                              # 右肘俯仰关节初始角度
22"right_wrist_roll_joint": 0.0,


self.mujoco_to_isaac_idx           = [   
0    0, # left_hip_pitch_joint
1    6, # right_hip_pitch_joint
2    12,# torso_joint 
3    1, # left_hip_roll_joint 
4    7, # right_hip_roll_joint
5    13,# left_shoulder_pitch_joint  
6    18,# right_shoulder_pitch_joint
7    2, # left_hip_yaw_joint   
8    8, # right_hip_yaw_joint 
9    14,# left_shoulder_roll_joint 
10   19,# right_shoulder_roll_joint  
11   3, # left_knee_joint 
12   9, # right_knee_joint
13   15,# left_shoulder_yaw_joint  
14   20,# right_shoulder_yaw_joint
15   4, # left_ankle_pitch_joint
16   10,# right_ankle_pitch_joint 
17   16,# left_elbow_joint 
18   21,# right_elbow_joint
19   5, # left_ankle_roll_joint
20   11,# right_ankle_roll_joint
21   17,# left_wrist_roll_joint
22   22,# right_wrist_roll_joint
]


self.isaac_to_mujoco_idx           = [      
0    0, # left_hip_pitch_joint 
1    3, # left_hip_roll_joint
2    7, # left_hip_yaw_joint
3    11,# left_knee_joint
4    15,# left_ankle_pitch_joint
5    19,# left_ankle_roll_joint
6    1, # right_hip_pitch_joint
7    4, # right_hip_roll_joint
8    8, # right_hip_yaw_joint
9    12,# right_knee_joint
10   16,# right_ankle_pitch_joint
11   20,# right_ankle_roll_joint
12   2, # torso_joint
13   5, # left_shoulder_pitch_joint
14   9, # left_shoulder_roll_joint
15   13,# left_shoulder_yaw_joint
16   17,# left_elbow_joint
17   21,# left_wrist_roll_joint
18   6, # right_shoulder_pitch_joint
19   10,# right_shoulder_roll_joint
20   14,# right_shoulder_yaw_joint
21   18,# right_elbow_joint
22   22,# right_wrist_roll_joint
]






"""
