import GymWrapper as gw
import optuna.visualization as vis
import optuna
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from GymWrapper import GymInterface


def tuning_hyperparam(trial):
    # Initialize the environment
    env = GymInterface()
    env.reset()
    # Define search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256])
    # Define the RL model
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=learning_rate,
                     gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, n_steps=SIM_TIME, verbose=0)
    # Train the model
    model.learn(total_timesteps=SIM_TIME*N_EPISODES)
    # Evaluate the model
    eval_env = GymInterface()
    mean_reward, _ = gw.evaluate_model(model, eval_env, N_EVAL_EPISODES)

    return -mean_reward  # Minimize the negative of mean reward


def run_optuna():
    # study = optuna.create_study( )
    study = optuna.create_study(direction='minimize')
    study.optimize(tuning_hyperparam, n_trials=N_TRIALS)

    # Print the result
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    # Visualize hyperparameter optimization process
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=['learning_rate', 'gamma']).show()
