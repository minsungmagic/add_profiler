import os
import shutil
from config_SimPy import *
import torch

# change the device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# RL algorithms
RL_ALGORITHM = "PPO"  # "DQN", "PPO"

ACTION_SPACE = [0, 1, 2, 3, 4, 5]

BUFFER_SIZE = 100000
BATCH_SIZE = 64  
LEARNING_RATE = 0.0001
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_STEPS = 10  
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

'''
# State space
STATE_RANGES = []
for i in range(len(I)):
    # Inventory level
    STATE_RANGES.append((0, INVEN_LEVEL_MAX))
    # Daily change for the on-hand inventory
    STATE_RANGES.append((-INVEN_LEVEL_MAX, INVEN_LEVEL_MAX))
# Remaining demand: Demand quantity - Current product level
STATE_RANGES.append((0, max(DEMAND_QTY_MAX, INVEN_LEVEL_MAX)))
'''
# Find minimum Delta
PRODUCT_OUTGOING_CORRECTION = 0
for key in P:
    PRODUCT_OUTGOING_CORRECTION = max(P[key]["PRODUCTION_RATE"] *
                                      max(P[key]['QNTY_FOR_INPUT_ITEM']), INVEN_LEVEL_MAX)
# maximum production

# Episode
N_EPISODES = 20 # 3000


def DEFINE_FOLDER(folder_name):
    file_list = os.listdir(folder_name)
    folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name.replace("\\", "/")


def save_path(path):

    # Create a new folder
    os.makedirs(path, exist_ok=True)
    return path


# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 15  # 50

# RL_Options
INTRANSIT = 1  # 0 Means False , 1 Means True
USE_CORRECTION = True
EXPERIMENT = False

# Evaluation
N_EVAL_EPISODES = 100  # 100

# Export files
DAILY_REPORT_EXPORT = False
STATE_TRAIN_EXPORT = False
STATE_TEST_EXPORT = False

# Define parent dir's path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# Define each dir's parent dir's path
tensorboard_folder = os.path.join(parent_dir, "tensorboard_log")
experiment_folder = os.path.join(parent_dir, "experiment_log")
result_csv_folder = os.path.join(parent_dir, "result_CSV")
STATE_folder = os.path.join(result_csv_folder, "state")
result_experiment = os.path.join(result_csv_folder, "Experiment_Result")
daily_report_folder = os.path.join(result_csv_folder, "daily_report")

# Define dir's path
TENSORFLOW_LOGS = DEFINE_FOLDER(tensorboard_folder)
if EXPERIMENT:
    EXPERIMENT_LOGS = DEFINE_FOLDER(experiment_folder)
'''
STATE = DEFINE_FOLDER(STATE_folder)
REPORT_LOGS = DEFINE_FOLDER(daily_report_folder)
GRAPH_FOLDER = DEFINE_FOLDER(graph_folder)
'''
if EXPERIMENT:
    RESULT_CSV_EXPERIMENT = save_path(result_experiment)
STATE = save_path(STATE_folder)
REPORT_LOGS = save_path(daily_report_folder)

# Makedir
'''
if os.path.exists(STATE):
    pass
else:
    os.makedirs(STATE)

if os.path.exists(REPORT_LOGS):
    pass
else:
    os.makedirs(REPORT_LOGS)
if os.path.exists(GRAPH_FOLDER):
    pass
else:
    os.makedirs(GRAPH_FOLDER)
'''
# Visualize_Graph
VIZ_INVEN_LINE = False
VIZ_INVEN_PIE = False
VIZ_COST_PIE = False
VIZ_COST_BOX = False

# Saved Model
SAVED_MODEL_PATH = os.path.join(parent_dir, "Saved_Model")
SAVE_MODEL = False
SAVED_MODEL_NAME = "DQN_MODEL_test_val"

# Load Model
LOAD_MODEL = False
LOAD_MODEL_NAME = "DQN_MODEL_SIM500"

# Non-stationary demand
mean_demand = 100
standard_deviation_demand = 20


# tensorboard --logdir="~\tensorboard_log"
# http://localhost:6006/