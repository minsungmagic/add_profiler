import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_RL import *
import environment as env
from log_SimPy import *
from log_RL import *
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from DQN import *
from PPO import *

class GymInterface(gym.Env):
    """
    A Gym environment interface for simulation using SimPy and Deep Q-Network.
    
    Attributes:
        writer: Tensorboard writer for logging.
        scenario: Scenario settings for demand and lead time.
        action_space: Action space for the environment.
        observation_space: Observation space for the environment.
        agent (DQNAgent): Reinforcement learning agent.
        buffer: Replay buffer.
    """
    def __init__(self):
        self.outer_end = False
        super(GymInterface, self).__init__()
        #self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
        if EXPERIMENT:
            self.scenario = {}
        else:
            self.scenario = {"DEMAND": DEMAND_SCENARIO,
                             "LEADTIME": LEADTIME_SCENARIO}

        self.shortages = 0
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.cur_episode = 1
        self.cur_outer_loop = 1  # Current outer loop
        self.cur_inner_loop = 1  # Current inner loop
        self.scenario_batch_size = 99999  # Initialize the scenario batch size

        # For functions that only work when testing the model
        self.model_test = False
        
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        os = []

        if RL_ALGORITHM == "DQN":
            action_space = []
            for i in range(len(I)):
                if I[i]["TYPE"] == "Material":
                    action_space.append(len(ACTION_SPACE))
            self.action_space = spaces.Discrete(len(ACTION_SPACE))
            # if self.scenario["Dist_Type"] == "UNIFORM":
            #    k = INVEN_LEVEL_MAX*2+(self.scenario["max"]+1)

            os = [
                INVEN_LEVEL_MAX * 2 + 1 for _ in range(len(I)+MAT_COUNT*INTRANSIT+1)]

            '''
            - Inventory Level of Product 
            - Inventory Level of WIP 
            - Inventory Level of Material 
            - Intransit Level of Material
            - Demand - Inventory Level of Product
            '''
            
            self.observation_space = spaces.MultiDiscrete(os)

            
        elif RL_ALGORITHM == "PPO":
            # Define action space
            actionSpace = []
            for i in range(len(I)):
                if I[i]["TYPE"] == "Material":
                    actionSpace.append(len(ACTION_SPACE))
            self.action_space = spaces.MultiDiscrete(actionSpace)
            # if self.scenario["Dist_Type"] == "UNIFORM":
            #    k = INVEN_LEVEL_MAX*2+(self.scenario["max"]+1)

            os = [
                INVEN_LEVEL_MAX * 2 + 1 for _ in range(len(I)+MAT_COUNT*INTRANSIT+1)]
            '''
            - Inventory Level of Product 
            - Inventory Level of WIP 
            - Inventory Level of Material 
            - Intransit Level of Material
            - Demand - Inventory Level of Product
            '''
            self.observation_space = spaces.MultiDiscrete(os)


    def reset(self):
        """
        Reset the environment for a new episode.
          
        Returns:
            state_real: The initial state of the environment.
        """

        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        
        # Initialize the simulation environment
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, LIST_DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I, self.scenario)
        env.update_daily_report(self.inventoryList)

        state_real = self.get_current_state()
    
        return state_real

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: The action chosen by the agent.
            
        Returns:
            next_state: The next state after the action.
            reward: The reward received.
            done: Whether the episode is finished.
            info: Additional information (if any).
        """
        
        if RL_ALGORITHM == "DQN":
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    I[_]["LOT_SIZE_ORDER"] = action  
                    break
        elif RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    # Set action as predicted value
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    i += 1
                
        # Run the simulation for 24 hours (until the next day)
        # Action append
        STATE_ACTION_REPORT_REAL[-1].append(action)
        self.simpy_env.run(until=self.simpy_env.now + 24)
        env.update_daily_report(self.inventoryList)
        
        # Capture the next state of the environment
        state_real = self.get_current_state()
        
        # Set the next state
        next_state = state_real
        
        # Calculate the total cost of the day
        env.Cost.update_cost_log(self.inventoryList)
        
        if PRINT_DAILY_EVENTS:
            cost = dict(DICT_DAILY_COST)
            
        # Cost Dict update
        for key in DICT_DAILY_COST.keys():
            self.cost_dict[key] += DICT_DAILY_COST[key]

        env.Cost.clear_cost()

        reward = -LIST_LOG_COST[-1]
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        if PRINT_DAILY_EVENTS:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "DQN":
                for _ in range(len(I)): 
                    if I[_]["TYPE"] == "Material":  
                        print(f"[Order Quantity for {I[_]['NAME']}] ", action)  
                        break  
            elif RL_ALGORITHM == "PPO":
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Material":
                        print(f"[Order Quantity for {I[_]['NAME']}] ", action)
                        break
                    
            # SimPy simulation print
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            for _ in cost.keys():
                print(_, cost[_])
            print("Total cost: ", -self.total_reward)
            print("[REAL_STATE for the next round] ",  [
                item-INVEN_LEVEL_MAX for item in next_state])

        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        
        if done == True:
            '''
            self.writer.add_scalar(
                "reward", self.total_reward, global_step=self.cur_episode)
            for cost_name, cost_value in self.cost_dict.items():
                self.writer.add_scalar(
                    cost_name, cost_value, global_step=self.cur_episode)
            self.writer.add_scalars(
                'Cost', self.cost_dict, global_step=self.cur_episode)
            '''
            print("Episode: ", self.cur_episode,
                  " / Total reward: ", self.total_reward,
                  " / Action: ", action)
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0
            self.cur_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)
        return next_state, reward, done, info

    def get_current_state(self):
        """
        Generate the current state representation for the agent.
     
        Returns:
            state: The current state, including on-hand inventory levels, intransit inventory levels and remaining demand.
        """
 
        state = []

        for id in range(len(I)):
            # append On_Hand_inventory
            state.append(
                LIST_LOG_STATE_DICT[-1][f"On_Hand_{I[id]['NAME']}"]+INVEN_LEVEL_MAX)
            
            # append In_Transit_inventory
            if INTRANSIT == 1:
                if I[id]["TYPE"] == "Material":
                    # append Intransition inventory
                    state.append(
                        LIST_LOG_STATE_DICT[-1][f"In_Transit_{I[id]['NAME']}"])

        # Append remaining demand
        state.append(I[0]["DEMAND_QUANTITY"] -
                     self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)
        STATE_ACTION_REPORT_REAL.append(
            [Item - INVEN_LEVEL_MAX for Item in state])
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        # 필요한 경우, 여기서 리소스를 정리
        pass

def evaluate_model(model, env, num_episodes):
    """
    Evaluate the trained model over a number of episodes.
    
    Args:
        model: The trained RL model.
        env (GymInterface): The Gym environment.
        num_episodes: Number of episodes for evaluation.
        
    Returns:
        mean_reward: Mean reward over episodes.
        std_reward: Standard deviation of rewards.
    """
    all_rewards = []  
    STATE_ACTION_REPORT_REAL.clear()
    ORDER_HISTORY = []
    Mat_Order = {}
    for mat in range(MAT_COUNT):
        Mat_Order[f"mat {mat}"] = []
    demand_qty = []
    onhand_inventory = []
    test_order_mean = []  # List to store average orders per episode
    for i in range(num_episodes):
        ORDER_HISTORY.clear()
        episode_inventory = [[] for _ in range(len(I))]
        LIST_LOG_DAILY_REPORTS.clear()  # Clear daily reports at the start of each episode
        obs = env.reset()  # Reset the environment to get initial observation
        episode_reward = 0  # Initialize reward for the episode
        env.model_test = True
        done = False  # Flag to check if episode is finished
        day = 1  # 차후 validaition끝나면 지울것
        while not done:
            for x in range(len(env.inventoryList)):
                episode_inventory[x].append(
                    env.inventoryList[x].on_hand_inventory)
            if RL_ALGORITHM == "DQN":
                action = model.select_action(obs, epsilon = max(0.1, 1.0 - N_EPISODES/500))
            elif RL_ALGORITHM == "PPO":
                action = model.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward  # Accumulate rewards
            ORDER_HISTORY.append(action)  # Log order history
            Mat_Order[f"mat 0"].append(action)
            # Mat_Order.append(I[1]["LOT_SIZE_ORDER"])
            demand_qty.append(I[0]["DEMAND_QUANTITY"])
            day += 1  # 추후 validation 끝나면 지울 것

        onhand_inventory.append(episode_inventory)
        all_rewards.append(episode_reward)  # Store total reward for episode
        # Function to visualize the environment

        # Calculate mean order for the episode
        order_mean = []
        for key in Mat_Order.keys():
            order_values = [item[0] for item in Mat_Order[key]]
            order_mean.append(sum(order_values) / len(order_values)) 
        test_order_mean.append(order_mean)
        COST_RATIO_HISTORY.append(env.cost_dict)

    Visualize_invens(onhand_inventory, demand_qty, Mat_Order, all_rewards)
    cal_cost_avg()
    if STATE_TEST_EXPORT:
        export_state("TEST")
    # Calculate mean reward across all episodes
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)  # Calculate standard deviation of rewards
    return mean_reward, std_reward  # Return mean and std of rewards


def cal_cost_avg():
    """
    Calculate and visualize the average cost over evaluation episodes.
    
    """
    cost_avg = {
        'Holding cost': 0,
        'Process cost': 0,
        'Delivery cost': 0,
        'Order cost': 0,
        'Shortage cost': 0
    }

    total_avg = []

    # Cal_cost_AVG
    for x in range(N_EVAL_EPISODES):
        for key in COST_RATIO_HISTORY[x].keys():
            cost_avg[key] += COST_RATIO_HISTORY[x][key]
        total_avg.append(sum(COST_RATIO_HISTORY[x].values()))
    for key in cost_avg.keys():
        cost_avg[key] = cost_avg[key]/N_EVAL_EPISODES
        
    # Visualize
    if VIZ_COST_PIE:
        plt.figure(figsize=(10, 5))
        plt.pie(cost_avg.values(), explode=[
                0.2 for x in range(5)], labels=cost_avg.keys(), autopct='%1.1f%%')
        path = os.path.join(GRAPH_LOG, 'COST_PI.png')
        plt.savefig(path)
        plt.show()
        plt.close()

    if VIZ_COST_BOX:
        plt.boxplot(total_avg)
        path = os.path.join(GRAPH_LOG, 'COST_BOX.png')
        plt.savefig(path)
        plt.show()
        plt.close()


def Visualize_invens(inventory, demand_qty, Mat_Order, all_rewards):
    """
    Visualize inventory, demand, and order history.
    
    Args:
        inventory: Inventory data over episodes.
        demand_qty: Demand quantities over episodes.
        Mat_Order: Order history for materials.
        all_rewards: Total rewards for episodes.

    """
    best_reward = -99999999999999
    best_index = 0
    for x in range(N_EVAL_EPISODES):
        if all_rewards[x] > best_reward:
            best_reward = all_rewards[x]
            best_index = x

    avg_inven = [[0 for _ in range(SIM_TIME)] for _ in range(len(I))]
    lable = []
    for id in I.keys():
        lable.append(I[id]["NAME"])

    if VIZ_INVEN_PIE:
        plt.figure(figsize=(10, 5))
        for x in range(N_EVAL_EPISODES):
            for y in range(len(I)):
                for z in range(SIM_TIME):
                    avg_inven[y][z] += inventory[x][y][z]

        plt.pie([sum(avg_inven[x])/N_EVAL_EPISODES for x in range(len(I))],
                explode=[0.2 for _ in range(len(I))], labels=lable, autopct='%1.1f%%')
        path = os.path.join(GRAPH_LOG, 'INVEN_PI.png')
        plt.savefig(path)
        plt.show()
        plt.close()

    if VIZ_INVEN_LINE:
        line_dict = {}
        writer = SummaryWriter(log_dir=GRAPH_LOG)
        plt.figure(figsize=(15, 5))
        # Inven Line
        for id in I.keys():
            # Visualize the inventory levels of the best episode
            plt.plot(inventory[best_index][id], label=lable[id])
            line_dict[lable[id]] = inventory[best_index][id]

        plt.plot(demand_qty[-SIM_TIME:], "y--", label="Demand_QTY")
        line_dict[f"Demand_QTY"] = demand_qty[-SIM_TIME:]
        # Order_Line
        for key in Mat_Order.keys():
            line_dict[f"ORDER {key}"] = Mat_Order[key][-SIM_TIME:]
            plt.plot(Mat_Order[key][-SIM_TIME:], label=f"ORDER {key}")
        plt.yticks(range(0, 21, 5))
        plt.legend(bbox_to_anchor=(1, 0.5))
        path = os.path.join(GRAPH_LOG, 'INVEN_LINE.png')
        plt.savefig(path)
        plt.show()
        plt.close()

        for day in range(SIM_TIME):
            temp_dict = {}
            for key, item in line_dict.items():
                temp_dict[key] = item[day]
            writer.add_scalars("Test_Line", temp_dict, global_step=day+1)


def export_state(Record_Type):
    """
    Export the state-action report as a CSV file.
    
    """
    state_real = pd.DataFrame(STATE_ACTION_REPORT_REAL)

    if Record_Type == 'TEST':
        state_real.dropna(axis=0, inplace=True)

    columns_list = []
    for id in I.keys():
        if I[id]["TYPE"] == 'Material':
            columns_list.append(f"{I[id]['NAME']}.InvenLevel")
            if INTRANSIT:
                columns_list.append(f"{I[id]['NAME']}.Intransit")
        else:
            columns_list.append(f"{I[id]['NAME']}.InvenLevel")
    columns_list.append("Remaining_Demand")
    columns_list.append("Action")
    '''
    for keys in I:
        columns_list.append(f"{I[keys]['NAME']}'s inventory")
        columns_list.append(f"{I[keys]['NAME']}'s Change")
    
    columns_list.append("Remaining Demand")
    columns_list.append("Action")
    '''
    state_real.columns = columns_list
    state_real.to_csv(f'{STATE}/STATE_ACTION_REPORT_REAL_{Record_Type}.csv')