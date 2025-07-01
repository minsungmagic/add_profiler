import os
import csv
import time
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *
import torch.profiler

def make_profiler_schedule(
    sim_step_per_episode: int,
    n_episodes: int,
    record_ratio: float = 0.1,
    record_per_episode: bool = True
):
    if record_per_episode:
        active = sim_step_per_episode
        repeat = int(n_episodes * record_ratio)
        return torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=active,
            repeat=repeat
        )
    else:
        total_steps = sim_step_per_episode * n_episodes
        total_record = int(total_steps * record_ratio)
        active = min(100, total_record)
        repeat = max(1, total_record // active)
        return torch.profiler.schedule(
            wait=10,
            warmup=5,
            active=active,
            repeat=repeat
        )

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

profiler = torch.profiler.profile(
    schedule=make_profiler_schedule(200, N_EPISODES, record_ratio=0.1, record_per_episode=True),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(TENSORFLOW_LOGS),
    record_shapes=True,
    with_stack=False,
    profile_memory=True,
    with_flops=True
)

N_MULTIPROCESS = 1

def build_model(env):
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
    return model

def simulation_worker(core_index, model_state_dict):
    env = GymInterface()
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=200, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(TENSORFLOW_LOGS, f"worker_{core_index}")),
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as sim_profiler:
        start_sim_time = time.time()
        state = env.reset()
        done = False
        episode_transitions = []
        episode_reward = 0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
            episode_reward += reward
            state = next_state
            sim_profiler.step()
        finish_sim_time = time.time()

    sim_time = finish_sim_time - start_sim_time
    return core_index, sim_time, finish_sim_time, episode_transitions, episode_reward

def process_transitions(transitions):
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

    episode_param_copy_times = []
    episode_sampling_times = []
    episode_waiting1_times = []
    episode_waiting2_times = []
    episode_transfer_times = []
    episode_total_learning_times = []
    episode_learning_times = []

    env_main = GymInterface()
    if LOAD_MODEL:
        model = build_model(env_main)
        model.policy.load_state_dict(
            torch.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME))
        )
        print(f"{LOAD_MODEL_NAME} loaded successfully")
    else:
        model = build_model(env_main)

    start_time = time.time()

    with profiler:
        while episode_counter < total_episodes:
            batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)

            start_copy = time.time()
            model_state_dict = model.policy.state_dict()
            param_copy = time.time() - start_copy
            episode_param_copy_times.append(param_copy)
            end_learning = 0

            tasks = [(i, model_state_dict) for i in range(batch_workers)]

            sampling_times = []
            transfer_times = []
            waiting1_times = []
            waiting2_times = []
            
            for core_index, sampling, finish_sim_time, transitions, episode_reward in pool.imap_unordered(worker_wrapper, tasks):
                receive_time = time.time()
                if end_learning == 0:
                    transfer = receive_time - finish_sim_time
                    waiting1 = 0
                else:
                    transfer = receive_time - end_learning
                    waiting1 = receive_time - finish_sim_time - transfer
                
                waiting1_times.append(waiting1)
                sampling_times.append(sampling)
                transfer_times.append(transfer)

                start_total_learn = time.time()
                states, actions, rewards, next_states, dones, log_probs = process_transitions([transitions])
                for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, log_probs):
                    model.store_transition((s, a, r, ns, d, lp))

                profiler.step()
                
                # total learning update
                model.update()
                end_learning = time.time()
                waiting2_times.append(end_learning)
                total_learn = end_learning - start_total_learn
                episode_total_learning_times.append(total_learn)
                
                # learning time
                learn = model.learn_time
                episode_learning_times.append(learn)

                episode_counter += 1
                main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter)
                main_writer.add_scalar("reward_average", episode_reward, episode_counter)

                print(
                    f"Worker {core_index} done — episode {episode_counter}: "
                    f"Copy {param_copy:.3f}s, Sampling {sampling:.3f}s, "
                    f"Transfer {transfer:.3f}s, Total_Learn {total_learn:.3f}s, Learn {learn:.3f}s"
                )

            waiting2_times = [max(waiting2_times) - x for x in waiting2_times]

            avg_sampling = sum(sampling_times) / len(sampling_times)
            avg_transfer = sum(transfer_times) / len(transfer_times)
            avg_waiting1 = sum(waiting1_times) / len(waiting1_times)
            avg_waiting2 = sum(waiting2_times) / len(waiting2_times)

            episode_sampling_times.append(avg_sampling)
            episode_transfer_times.append(avg_transfer)
            episode_waiting1_times.append(avg_waiting1)
            episode_waiting2_times.append(avg_waiting2)

    # experiment summary
    total_time = (time.time() - start_time) / 60
    final_avg_param_copy = sum(episode_param_copy_times) 
    final_avg_sampling = sum(episode_sampling_times)
    final_waiting1_tIme = sum(episode_waiting1_times)
    final_waiting2_tIme = sum(episode_waiting2_times)
    final_avg_transfer = sum(episode_transfer_times)
    final_avg_total_learning = sum(episode_total_learning_times)
    final_avg_learning = sum(episode_learning_times)
    print(
        f"\n[Experiment Summary] "
        f"Copy {final_avg_param_copy:.6f}s | "
        f"Sampling {final_avg_sampling:.6f}s | "\
        f"Waiting1 {final_waiting1_tIme:.6f}s | "
        f"Transfer {final_avg_transfer:.6f}s | "
        f"Waiting2 {final_waiting2_tIme:.6f}s | "
        f"Total_Learn {final_avg_total_learning:.6f}s | "
        f"Learn {final_avg_learning:.6f}s | "
        f"Total {total_time:.6f}min\n"
    )

    # Assuming the variables are already calculated as in your summary
    data = {
        'Copy': final_avg_param_copy,
        'Sampling': final_avg_sampling,
        'Waiting1': final_waiting1_tIme,
        'Transfer': final_avg_transfer,
        'Waiting2': final_waiting2_tIme,
        'Total_Learn': final_avg_total_learning,
        'Learn': final_avg_learning,
        'Total': total_time
    }

    # Save to CSV
    with open(f"{N_MULTIPROCESS}core_test1_누적.csv", 'w', newline='') as csvfile:
        fieldnames = ['Copy', 'Sampling', 'Waiting1','Transfer', 'Waiting2','Total_Learn', 'Learn', 'Total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({key: f'{value:.6f}' for key, value in data.items()})
    print(f"[Profiler LogDir] → {TENSORFLOW_LOGS}")

    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    pool.close()
    pool.join()
