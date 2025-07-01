import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
from DQN import *
from PPO import *

def build_model():
    if RL_ALGORITHM == "DQN":
        model = DQNAgent(state_dim=env.observation_space.shape[0], 
                         action_dim=env.action_space.n, 
                         buffer_size=BUFFER_SIZE, lr=LEARNING_RATE, gamma=GAMMA)
    elif RL_ALGORITHM == "PPO":
        model = PPOAgent(state_dim=env.observation_space.shape[0], 
                        action_dims=env.action_space.nvec,  # Adjusted for MultiDiscrete
                        lr=LEARNING_RATE, 
                        gamma=GAMMA,
                        clip_epsilon=CLIP_EPSILON,
                        update_steps=UPDATE_STEPS)  
    return model

# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Run hyperparameter optimization if enabled
if OPTIMIZE_HYPERPARAMETERS:
    ht.run_optuna()
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes ")
else:
    if LOAD_MODEL:
        if RL_ALGORITHM == "DQN":
            model = DQN.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)
        elif RL_ALGORITHM == "PPO":
            model = PPOAgent(state_dim=env.observation_space.shape[0], 
                             action_dims=env.action_space.nvec, 
                             lr=LEARNING_RATE, gamma=GAMMA)
            model.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME))
        print(f"{LOAD_MODEL_NAME} is loaded successfully")  
    else:
        model = build_model()
        
        if RL_ALGORITHM == "DQN":
            for episode in range(N_EPISODES):
                state = env.reset()
                done = False
                while not done:
                    action = model.select_action(state, epsilon=max(0.1, 1.0 - episode/500))
                    next_state, reward, done, _ = env.step(action)                
                    model.buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    model.update(batch_size=BATCH_SIZE)
        
        elif RL_ALGORITHM == "PPO":
            for episode in range(N_EPISODES):
                state = env.reset()
                done = False
                while not done:
                    action, log_prob = model.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    model.store_transition((state, action, reward, next_state, done, log_prob))
                    state = next_state
                model.update()
        
        if SAVE_MODEL:
            model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
            print(f"{SAVED_MODEL_NAME} is saved successfully")

        if STATE_TRAIN_EXPORT:
            gw.export_state('TRAIN')

    training_end_time = time.time()

    # Evaluate the trained model
    mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
    print(f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Compute timing details
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes ",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes ",
          f"Test time: {(end_time - training_end_time)/60:.2f} minutes")

# Optionally render the environment
env.render()