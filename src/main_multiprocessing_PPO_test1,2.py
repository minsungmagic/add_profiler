import os
import time
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
N_MULTIPROCESS = 5

def build_model(env):
    """
    Build a PPO model using environment info.
    """
    state_dim = len(env.reset())
    action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]
    model = PPOAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        clip_epsilon=CLIP_EPSILON,
        update_steps=UPDATE_STEPS
    )

    # validation) Verify training model device assignment
    #print(f"[Main] Training model on device: {model.device}")
    
    return model

def simulation_worker(core_index, model_state_dict):
    """
    Simulates one episode using a local copy of the PPO model.

    Args:
        core_index (int): Index of the current process.
        model_state_dict (dict): State dict from the main PPO model.

    Returns:
        tuple: (core_index, sim_time, finish_time, transitions, episode_reward)
    """
    env = GymInterface()
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)
    
    # validation) Verify one inference model per core
    #print(f"[Worker {core_index} PID {os.getpid()}] Inference model id: {id(agent.policy)}")

    # validation) Verify inference models are loaded on CPU/GPU
    #print(f"[Worker {core_index} | PID {os.getpid()}] Inference model.device: {agent.device}")
    #print(f"[Worker {core_index}] first_param.device: {next(agent.policy.parameters()).device}")
    
    start_sim_time = time.time()
    state = env.reset()
    
    # validation) Verify simulator runs on CPU
    #print(f"[Worker {core_index}] state type: {type(state)}")  # must be list or numpy
    
    done = False
    episode_transitions = []
    episode_reward = 0
    while not done:
        action, log_prob = agent.select_action(state)  
        
        # validation) Verify where select_action() is executed (inference model location)
        #print(f"[Worker {core_index}] log_prob.device: {log_prob.device}")
        
        next_state, reward, done, _ = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state
    finish_sim_time = time.time()
    sim_time = finish_sim_time - start_sim_time

    return core_index, sim_time, finish_sim_time, episode_transitions, episode_reward

def process_transitions(transitions):
    """
    Flattens a list of per-worker transition lists.
    """
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    for worker_transitions in transitions:
        for (s, a, r, ns, d, lp) in worker_transitions:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            log_probs.append(lp)
    return states, actions, rewards, next_states, dones, log_probs

def worker_wrapper(args):
    return simulation_worker(*args)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    total_episodes = N_EPISODES
    episode_counter = 0

    # timing records
    episode_param_copy_times = []
    episode_sampling_times = []
    episode_transfer_times = []
    episode_total_learning_times = []
    episode_learning_times = []

    # build or load main model on GPU
    env_main = GymInterface()
    if LOAD_MODEL:
        model = build_model(env_main)
        model.policy.load_state_dict(
            torch.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME))
        )
        print(f"{LOAD_MODEL_NAME} loaded successfully")
    else:
        model = build_model(env_main)
        
        # validation) Total parameter count
        #total_params = sum(p.numel() for p in model.policy.parameters())
        #print(f"Total parameters: {total_params}")

        # validation) Trainable parameter count
        #trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        #print(f"Trainable parameters: {trainable_params}")

        # validation) Verify single training model
        #print(f"[Main PID {os.getpid()}] Training model id: {id(model.policy)}")

    start_time = time.time()

    while episode_counter < total_episodes:
        batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
        # measure parameter copy time
        start_copy = time.time()
        model_state_dict = model.policy.state_dict()
        param_copy = time.time() - start_copy
        episode_param_copy_times.append(param_copy)

        tasks = [(i, model_state_dict) for i in range(batch_workers)]

        sampling_times = []
        transfer_times = []

        # independent buffer: process as workers finish
        for core_index, sampling, finish_sim_time, transitions, episode_reward in pool.imap_unordered(worker_wrapper, tasks): 
            
            # validation) Check independent buffer
            #print(f"[Main] Got result from worker {core_index} at {time.time():.3f}")
            
            receive_time = time.time()
            transfer = receive_time - finish_sim_time

            sampling_times.append(sampling)
            transfer_times.append(transfer)

            # store transitions
            start_total_learn = time.time()
            states, actions, rewards, next_states, dones, log_probs = process_transitions([transitions])
            for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, log_probs):
                model.store_transition((s, a, r, ns, d, lp))

            # total learning update
            model.update()
            total_learn = time.time() - start_total_learn
            episode_total_learning_times.append(total_learn)
            
            # learning time
            learn = model.learn_time
            episode_learning_times.append(learn)
            
            # tensorboard 
            episode_counter += 1
            main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter)
            main_writer.add_scalar("reward_average", episode_reward, episode_counter)

            print(
                f"Worker {core_index} done — episode {episode_counter}: "
                f"Copy {param_copy:.3f}s, Sampling {sampling:.3f}s, "
                f"Transfer {transfer:.3f}s, Total_Learn {total_learn:.3f}s, Learn {learn:.3f}s"
            )

        avg_sampling = sum(sampling_times) / len(sampling_times)
        avg_transfer = sum(transfer_times) / len(transfer_times)

        episode_sampling_times.append(avg_sampling)
        episode_transfer_times.append(avg_transfer)

    # experiment summary
    total_time = (time.time() - start_time) / 60
    final_avg_param_copy = sum(episode_param_copy_times) / len(episode_param_copy_times)
    final_avg_sampling = sum(episode_sampling_times) / len(episode_sampling_times)
    final_avg_transfer = sum(episode_transfer_times) / len(episode_transfer_times)
    final_avg_total_learning = sum(episode_total_learning_times) / len(episode_total_learning_times)
    final_avg_learning = sum(episode_learning_times)/len(episode_learning_times)

    print(
        f"\n[Experiment Summary] "
        f"Copy {final_avg_param_copy:.6f}s | "
        f"Sampling {final_avg_sampling:.6f}s | "
        f"Transfer {final_avg_transfer:.6f}s | "
        f"Total_Learn {final_avg_total_learning:.6f}s | "
        f"Learn {final_avg_learning:.6f}s | "
        f"Total {total_time:.6f}min\n"
    )

    pool.close()
    pool.join()