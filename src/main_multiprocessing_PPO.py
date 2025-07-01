import GymWrapper as gw
import os
import time
import multiprocessing
import math
import torch
from GymWrapper import GymInterface 
from PPO import PPOAgent
from config_RL import *
from torch.utils.tensorboard import SummaryWriter

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

N_MULTIPROCESS = 5

def build_model(env):
    """
    Build and return a PPOAgent model using the environment's state dimension and MAT_COUNT.
    
    Args:
        env (GymInterface): The Gym environment instance.
        
    Returns:
        PPOAgent: A PPO agent instance initialized with proper hyperparameters.
    """
    state_dim = len(env.reset())
    action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]  # MultiDiscrete
    model = PPOAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        clip_epsilon=CLIP_EPSILON,
        update_steps=UPDATE_STEPS
    )
    return model

def simulation_worker(core_index, model_state_dict):
    """
    Run a single episode in a worker process and return the transitions and total reward.
    
    Args:
        core_index: The index of the worker process.
        model_state_dict: The state dictionary of the main model.
        
    Returns:
        tuple: (core_index, episode_transitions, episode_reward)
    """
    env = GymInterface()
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)
    
    state = env.reset()
    done = False
    episode_transitions = []
    episode_reward = 0
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state
    return (core_index, episode_transitions, episode_reward)

def process_transitions(transitions):
    """
    Combine and unpack transition data collected from a worker.
    
    Args:
        transitions (list): A list of transition lists from one or more workers.
    
    Returns:
        tuple: Separate lists for states, actions, rewards, next_states, dones, and log_probs.
    """
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    for worker_transitions in transitions:
        for tr in worker_transitions:
            states.append(tr[0])
            actions.append(tr[1])
            rewards.append(tr[2])
            next_states.append(tr[3])
            dones.append(tr[4])
            log_probs.append(tr[5])
    return states, actions, rewards, next_states, dones, log_probs

def worker_wrapper(args):
    """
    Wrapper function for simulation_worker to unpack arguments.
    
    Args:
        args (tuple): A tuple containing (core_index, model_state_dict)
        
    Returns:
        tuple: The result of simulation_worker.
    """
    return simulation_worker(*args)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)
    
    total_episodes = N_EPISODES  
    episode_counter = 0
    
    core_rewards = {i: [] for i in range(N_MULTIPROCESS)}
    
    env_main = GymInterface()
    if LOAD_MODEL:
        state_dim = len(env_main.reset())
        action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]
        model = PPOAgent(
            state_dim=state_dim,
            action_dims=action_dims,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            clip_epsilon=CLIP_EPSILON,
            update_steps=UPDATE_STEPS
        )
        model.policy.load_state_dict(torch.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME)))
        print(f"{LOAD_MODEL_NAME} loaded successfully")
    else:
        model = build_model(env_main)
    
    start_time = time.time()
    
    while episode_counter < total_episodes:
        batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
        model_state_dict = model.policy.state_dict()
        
        tasks = [(i, model_state_dict) for i in range(batch_workers)]
        
        # Use imap_unordered to retrieve results as soon as they are ready (FIFO order)
        for result in pool.imap_unordered(worker_wrapper, tasks):  # imap_unordered: 워커들의 결과를 먼저 끝난 순서대로 반환환
            core_index, transitions, episode_reward = result
            global_episode = episode_counter + 1
            main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, global_step=global_episode)
            
            states, actions, rewards, next_states, dones, log_probs = process_transitions([transitions])
            for j in range(len(states)):
                model.store_transition((states[j], actions[j], rewards[j], next_states[j], dones[j], log_probs[j]))
            
            model.update()
            episode_counter += 1
            print(f"Completed {episode_counter} / {total_episodes} episodes.")
            
            main_writer.add_scalar("reward_average", episode_reward, global_step=global_episode)
    
    if SAVE_MODEL:
        model_path = os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME)
        torch.save(model.policy.state_dict(), model_path)
        print(f"{SAVED_MODEL_NAME} saved successfully")
    
    end_time = time.time()
    computation_time = (end_time - start_time) / 60
    print(f"Total computation time: {computation_time:.2f} minutes")
    
    pool.close()
    pool.join()