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
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg( 
        pos=(0.0, 0.0, 0.752),
        joint_pos={      
            "left_hip_pitch_joint": 0.3,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": -0.6,
            "left_ankle_pitch_joint": -0.3,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.3,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.6,
            "right_ankle_pitch_joint": 0.3,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 1.3,
            "left_shoulder_yaw_joint": -0.6,
            "left_elbow_joint": 0.3,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -1.3,
            "right_shoulder_yaw_joint": 0.6,
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
                ".*_hip_pitch_joint":  50,
                ".*_hip_roll_joint":   50,
                ".*_hip_yaw_joint":    30,
                ".*_knee_joint":       50,
            },
            velocity_limit_sim={         
                ".*_hip_pitch_joint":  10,
                ".*_hip_roll_joint":   10,
                ".*_hip_yaw_joint":    10,
                ".*_knee_joint":       10,
            },
            stiffness={                                                 
                ".*_hip_pitch_joint":  100,
                ".*_hip_roll_joint":   100,
                ".*_hip_yaw_joint":    30,
                ".*_knee_joint":       100,

            },
            damping={                                                    
                ".*_hip_pitch_joint":  3,
                ".*_hip_roll_joint":   3,
                ".*_hip_yaw_joint":    2,
                ".*_knee_joint":       3,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[                                              
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={                                              
                ".*_ankle_pitch_joint":  15,
                ".*_ankle_roll_joint":   12,
            },
            velocity_limit_sim={                                            
                ".*_ankle_pitch_joint":  10,
                ".*_ankle_roll_joint":   10,
            },
            stiffness={                                                     
                ".*_ankle_pitch_joint":  30,
                ".*_ankle_roll_joint":   20,
            },
            damping={                                                      
                ".*_ankle_pitch_joint":  1.5,
                ".*_ankle_roll_joint":   1,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "torso": DelayedPDActuatorCfg(
            joint_names_expr=[
                "torso_joint",
            ],
            effort_limit_sim={                                            
                "torso_joint":           20,
            },
            velocity_limit_sim={                                           
                "torso_joint":           10,
            },
            stiffness={                                                   
                "torso_joint":           20,
            },
            damping={                                                     
                "torso_joint":           1.5,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[                                             
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={                                            
                ".*_shoulder_pitch_joint":  15,
                ".*_shoulder_roll_joint":   12,
                ".*_shoulder_yaw_joint":    12,
                ".*_elbow_joint":           12,
                ".*_wrist_roll_joint":      12,
            },
            velocity_limit_sim={                                           
                ".*_shoulder_pitch_joint":  10,
                ".*_shoulder_roll_joint":   10,
                ".*_shoulder_yaw_joint":    10,
                ".*_elbow_joint":           10,
                ".*_wrist_roll_joint":      10,
            },
            stiffness={                                                   
                ".*_shoulder_pitch_joint":  15,
                ".*_shoulder_roll_joint":   15,
                ".*_shoulder_yaw_joint":    10,
                ".*_elbow_joint":           10,
                ".*_wrist_roll_joint":      10,
            },
            damping={                                                     
                ".*_shoulder_pitch_joint":  1,
                ".*_shoulder_roll_joint":   1,
                ".*_shoulder_yaw_joint":    0.7,
                ".*_elbow_joint":           0.7,
                ".*_wrist_roll_joint":      0.7,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)
