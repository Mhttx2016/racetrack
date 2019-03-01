import numpy as np 
from pprint import pprint
from tqdm import tqdm
import os
from datetime import datetime
import pickle, turtle
import pdb

MAP_WIDTH = 17
MAP_HEIGHT = 32

#the row and column index of left down corner and right up corner of rectangle area, 
# e.g. (row_left_down , col_left_down, row_right_up(exclude), col_right_up(exclude))
START_AREA = (0, 3, 1, 9)
VALID_AREA = [(1, 3, 3, 9), 
              (3, 2, 10, 9), 
              (10, 1, 18, 9), 
              (18, 0, 25, 9), 
              (25, 0, 26, 10),
              (26, 0, 28, 16),
              (28, 1, 29, 16),
              (29, 2, 31, 16),
              (31, 3, 32, 16)]
FINISH_AREA = (26, 16, 32, 17)
# 1 for start line, 0 for blank area, 2 for valid area, 3 for finish line
MAP = np.zeros(shape=(MAP_HEIGHT, MAP_WIDTH), dtype=np.int32)
MAP[START_AREA[0]:START_AREA[2], START_AREA[1]:START_AREA[3]] = 1
MAP[FINISH_AREA[0]:FINISH_AREA[2], FINISH_AREA[1]:FINISH_AREA[3]] = 3
for area in VALID_AREA:
    MAP[area[0]:area[2], area[1]:area[3]] = 2

ACTIONS = [-1, 0, 1]

def take_action(state, action, noise_probability=None):
    '''take action from state
    @param state (tuple): velocity of horizontal and vertical direction, and position. e.g.(vx, vy, x, y)
    @param action (tiple): velocity changes of horizontal and vertical direction. e.g. (ax, ay)
    @param noise_probability (None or scale): To make the task more challenging, 
            with probability noise_probability at each time step the velocity increments 
            are both zero, independently of the intended increments. None for not adding noise

    @return new_state (tuple): state after taking action
    @return reward (scal): reward for this step
    @return terminal (bool): whether the episode terminate
    '''
    # pdb.set_trace()
    vx, vy, x, y = state
    # add nosie
    if noise_probability is not None:
        if np.random.rand() < 0.1 and not (vx==0 and vy==0):
            action = (0, 0)

    # update velocity
    
    ax, ay = action
    new_vx = vx + ax
    new_vy = vy + ay
    assert new_vx >= 0 and new_vx <= 5
    assert new_vy >= 0 and new_vy <= 5
    assert not (new_vx == 0 and new_vy == 0)

    # check_intersect
    area_flag = check_intersect(x, y, new_vx, new_vy)
    
    if area_flag == 3: # finish
        new_state, reward, terminal = None, 0, True
    elif area_flag == 0: # back start line
        new_state, reward, terminal = get_state_in_start_line(), -1, False
    else:
        new_state, reward, terminal = (new_vx, new_vy, x+new_vx, y+new_vy), -1, False
    return new_state, reward, terminal

def get_state_in_start_line():
    '''get random state in start line 
    @return state in start line 
    '''
    return (0, 0, np.random.randint(low=START_AREA[1], high=START_AREA[3]), 0)

def get_legal_actions(state):
    '''get all posibile actions from state
    @param state (tuple): (vx, vy, x, y)
    @return legal_actions (list of tuples): list of legal acgtions from state
    '''
    legal_actions = []
    for ax in [-1, 0, 1]:
        for ay in [-1, 0, 1]:
            new_vx = state[0] + ax
            new_vy = state[1] + ay
            if new_vx >= 0 and new_vx <= 5 and \
                new_vy >= 0 and new_vy <= 5 and \
                not (new_vx== 0 and new_vy == 0):
                legal_actions.append((ax, ay))
    return legal_actions

def random_policy(state):
    '''random policy
    @param state (tuple): (vx, vy, x, y)
    @return action (tuple): a selected action (ax, ay)
    @return probability: the probability of selected action from state 
    '''
    legal_actions = get_legal_actions(state)
    return legal_actions[np.random.randint(0, len(legal_actions))], 1.0 / len(legal_actions)

def initialize_q_value():
    '''initialize q_value
    '''
    q_value = np.emmpty(shape=(6, 6, MAP_WIDTH, MAP_HEIGHT, 3, 3)) # q_value[vx][vy][x][y][ax][ay]
    q_value.fill(-float('inf'))
    return q_value

def check_intersect(x, y, vx, vy):
    '''check the intersect of moving from position (x, y) to position (x+vx, y+vy)
    @return area_flag: 0 for blank area(out and back to start line), 
    3 for finish(episode terminate), 2 for valid area(episode continue)
    '''
    # pdb.set_trace()
    new_x, new_y = x, y
    for _ in range(vx):
        new_x += 1
        if new_x < 0 or new_x > MAP_WIDTH - 1: # out
            return 0
        elif MAP[new_y][new_x] == 3: # finish
            # print('finish')
            return 3
        elif MAP[new_y][new_x] == 0: # out
            return 0
    for _ in range(vy):
        new_y += 1
        if new_y < 0 or new_y > MAP_HEIGHT - 1: # out
            return 0
        elif MAP[new_y][new_x] == 3: # finish
            # print('finish')
            return 3
        elif MAP[new_y][new_x] == 0: # out
            return 0
    return 2 # valid area

def play(policy, initial_state):
    '''generate an episode from initial_state using given policy
    @param policy: a function given a state that return an action and probability of the action
    @param initial_state: the initial state of the episode
    @return trajectories: a list of tuples (state, action, reward)
    '''
    state = initial_state
    trajectories = []
    while True:
        action, _ = policy(state)
        last_state = state
        state, reward, terminal = take_action(last_state, action, noise_probability=0.1)
        if terminal:
            trajectories.append((last_state, action, reward))
            break
        trajectories.append((last_state, action, reward))
    return trajectories

def get_action_from_q_value(state, q_value):
    vx, vy, x, y = state
    this_state_value = q_value[vx][vy][x][y]
    indexes = np.where(q_value[vx][vy][x][y]==np.max(q_value[vx][vy][x][y]))
    choosed = np.random.choice(indexes[0].shape[0])
    action = (ACTIONS[indexes[0][choosed]], ACTIONS[indexes[1][choosed]])
    # print(state, action)
    return action

def off_policy_mc_control(episodes, gamma):
    '''off policy MC control incremental implementation
    '''

    def initilize():
        '''initilize q_value and cumulative_weights
        '''
        q_value = np.empty(shape=(6, 6, MAP_WIDTH, MAP_HEIGHT, 3, 3)) # q_value[vx][vy][x][y][ax][ay]
        q_value.fill(-float('inf'))
        for vx in range(6):
            for vy in range(6):
                for x in range(MAP_WIDTH):
                    for y in range(MAP_HEIGHT):
                        possibile_actions = get_legal_actions(state=(vx, vy, x, y))
                        for action in possibile_actions:
                            q_value[vx][vy][x][y][action[0]][action[1]] = 0
        cumulative_weights = np.zeros(shape=q_value.shape, dtype=np.int32)
        target_policy = [[[[get_action_from_q_value(state=(vx, vy, x, y), q_value=q_value) for y in range(MAP_HEIGHT)] \
                            for x in range(MAP_WIDTH)] \
                            for vy in range(6)] \
                            for vx in range(6)] 

        return q_value, cumulative_weights, target_policy

    q_value, cumulative_weights, target_policy = initilize()
    for i in tqdm(range(episodes)):
        initial_state = get_state_in_start_line()
        trajectories = play(policy=random_policy, initial_state=initial_state)
        # print('='*50)
        # print(trajectories[-1])
        g = 0
        w = 1
        for t, traj in enumerate(reversed(trajectories)):
            g = gamma * g + traj[2]
            action = traj[1]
            vx, vy, x, y = traj[0]
            cumulative_weights[vx][vy][x][y][action[0]][action[1]] += w
            q_value[vx][vy][x][y][action[0]][action[1]] += \
                    w / cumulative_weights[vx][vy][x][y][action[0]][action[1]] * \
                    (g - q_value[vx][vy][x][y][action[0]][action[1]])
            target_policy[vx][vy][x][y] = get_action_from_q_value(state=(vx, vy, x, y), q_value=q_value)
            
            if action != target_policy[vx][vy][x][y]:
                break
            _, behavior_action_probability = random_policy(state=(vx, vy, x, y))
            w /= behavior_action_probability
    return q_value, target_policy

def visualize(q_value):
    state = get_state_in_start_line()
    trajectories = []
    while True:
        # pdb.set_trace()
        action = get_action_from_q_value(state, q_value)
        last_state = state
        state, reward, terminal = take_action(last_state, action, noise_probability=0.1)
        if terminal:
            break
        trajectories.append((last_state, action, reward))

    for traj in trajectories:
        vx, vy, x, y = traj[0]
        turtle.speed(vx)
        turtle.right(90)
        turtle.forward(vx)

        turtle.speed(vy)
        turtle.left(90)
        trutle.forward(vy)



if __name__ == '__main__':
    q_value, target_policy = off_policy_mc_control(episodes=10000, gamma=1)
    now_str = str(datetime.now())[:-7].replace('-', '_').replace(':', '_').replace(' ', '_')
    model_dir = os.path.join('racetrack_result', now_str)
    os.mkdir(model_dir)
    with open(os.path.join(model_dir, 'policy.pkl'), 'wb') as f:
        pickle.dump(target_policy, f)
    np.save(os.path.join(model_dir, 'q_value.npy'), q_value)

    # q_value = np.load(os.path.join('racetrack_result', '2019_03_01_20_27_06', 'q_value.npy'))
    # visualize(q_value)
