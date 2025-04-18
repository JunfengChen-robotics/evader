
'''
All hyperparameters and settings for the NewHideAttack model
All settings are stored in a dictionary called 'setting'
All parameters are shared in the NewHideAttack model
'''
# ========================
# METHOD SWITCH
# ========================
GLOBAL_ESCAPE = True # True: global escape, False: local escape

# ========================
# common parameters
# ========================
DISCRETE_SIZE = 0.1
ONLINE_EX_TIME = 1
GOAL_TOLERANCE = 0.05


# ========================
# evader parameters
# ========================
PRED_STEP = 5
NP_SEED = 3
EVADER_COEFF = 3.0
SAMPLE_NUM = 8
EVADER_MODE = "RANDOM_TRAVEL"
EVADER_POLICY = "RETREAT"
# EVADER_POLICY = "SURROUND"
GOAL_SELECT = "QUICK"
NODE_COEFF1 = 0.3
NODE_COEFF2 = 0.5
NODE_COEFF3 = 0.7


# ========================
# Gates set parameters
# ========================
GATES_CHOOSE_NUM = 2
TOTAL_GATES_NUM = 4
HIDER_GATE_TOLERANCE = 4
DISTANCE_COEFF1 = 5.0
DISTANCE_COEFF2 = 1.2
DISTANCE_COEFF3 = 5.0
DISTANCE_COEFF4 = 0.5
# GATES_FILTER = "SIMILAR"
GATES_FILTER = "NORMAL"

# ========================
# convergence parameters
# ========================

DECAY_RATE = 0.1


# ========================
# Cost map parameters
# ========================
MAIN_BASE = 100  # Larger one: for cases need influence larger area
MAIN_SIGMA = 5

ASSIST_BASE = 20  # Smaller one
ASSIST_SIGMA = 3
ASSIST_SIGMA_P = 3 # 3 for capture_range 0.4, 5 for capture_range 0.8

BOUNDARY_BASE = 100
BOUNDARY_SIGMA = 0.1

