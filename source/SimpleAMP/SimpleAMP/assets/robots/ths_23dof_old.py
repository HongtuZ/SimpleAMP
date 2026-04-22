import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

THS23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/robot/hongtu/SimpleAMP/robot_assets/ths_23dof/usd/ths_23dof.usd",
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
        pos=(0.0, 0.0, 0.752),
        joint_pos={
            "left_hip_pitch_joint": 0.2,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": -0.5,
            "left_ankle_pitch_joint": -0.3,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.2,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.5,
            "right_ankle_pitch_joint": 0.3,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 1.4,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.3,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -1.4,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.3,
            "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 30,
                ".*_hip_roll_joint": 30,
                ".*_hip_yaw_joint": 20,
                ".*_knee_joint": 30,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 10,
                ".*_hip_roll_joint": 10,
                ".*_hip_yaw_joint": 10,
                ".*_knee_joint": 10,
            },
            stiffness={
                ".*_hip_pitch_joint": 60,
                ".*_hip_roll_joint": 60,
                ".*_hip_yaw_joint": 30,
                ".*_knee_joint": 60,
            },
            damping={
                ".*_hip_pitch_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_yaw_joint": 1.5,
                ".*_knee_joint": 2.5,
            },
            armature=0.02,
            min_delay=0,
            max_delay=2,
        ),
        "feet_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 10,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 10,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20,
            },
            damping={
                ".*_ankle_pitch_joint": 1.2,
            },
            armature=0.0042,
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
                ".*_ankle_roll_joint": 5,
            },
            damping={
                ".*_ankle_roll_joint": 0.3,
            },
            armature=0.001,
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
                "torso_joint": 20,
            },
            damping={
                "torso_joint": 1.0,
            },
            armature=0.012,
            min_delay=0,
            max_delay=2,
        ),
        "arms_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 5,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 10,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 30,
            },
            damping={
                ".*_shoulder_pitch_joint": 1.5,
            },
            armature=0.0042,
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
                ".*_shoulder_roll_joint": 5,
                ".*_shoulder_yaw_joint": 5,
                ".*_elbow_joint": 5,
            },
            velocity_limit_sim={
                ".*_shoulder_roll_joint": 10,
                ".*_shoulder_yaw_joint": 10,
                ".*_elbow_joint": 10,
            },
            stiffness={
                ".*_shoulder_roll_joint": 30,
                ".*_shoulder_yaw_joint": 10,
                ".*_elbow_joint": 10,
            },
            damping={
                ".*_shoulder_roll_joint": 1.5,
                ".*_shoulder_yaw_joint": 0.6,
                ".*_elbow_joint": 0.6,
            },
            armature=0.001,
            min_delay=0,
            max_delay=2,
        ),
        "arms_end": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": 1.0,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": 10,
            },
            stiffness={
                ".*_wrist_roll_joint": 0.5,
            },
            damping={
                ".*_wrist_roll_joint": 0.05,
            },
            armature=0.001,
            min_delay=0,
            max_delay=2,
        ),
    },
)
