#######################################################################
# Copyright (C)                                                       #
# 2019 Mhttx(mhttxgm@gmail.com)                                       #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np 
from pprint import pprint
from tqdm import tqdm
import os
from copy import copy
from datetime import datetime
import pickle, turtle
import pdb
import time 

MAP_WIDTH = 17
MAP_HEIGHT = 32
MIN_REWARD = -(MAP_HEIGHT + MAP_WIDTH)
FINISH_REWARD = 100
OUT_REWARD = 0.15 * MIN_REWARD

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

# for turtle graphic visulizing
COORDINATE_UNIT = 22
COORDINATE_ORIGIN = (-188 + COORDINATE_UNIT // 2, -360 + COORDINATE_UNIT // 2)

turtle.bgpic('bkg0.png')
turtle.penup()
turtle.pencolor('red')
turtle.setpos(COORDINATE_ORIGIN[0], COORDINATE_ORIGIN[1])
turtle.pendown()

def check_MAP():
    '''visualize the map to check the corretness
    '''
    color_map = ['white', 'red', 'black', 'green']
    for i in range(MAP_HEIGHT):
        for j in range(MAP_WIDTH):
            turtle.penup()
            turtle.setpos(j*COORDINATE_UNIT+COORDINATE_ORIGIN[0], i* COORDINATE_UNIT+COORDINATE_ORIGIN[1])
            turtle.pendown()
            turtle.pencolor(color_map[MAP[i][j]])
            turtle.dot(10)
    time.sleep(10)

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

def take_action(state, action, noise_probability=None):
    '''take action from state
    @param state (tuple): velocity of horizontal and vertical direction, and position. e.g.(vx, vy, x, y)
    @param action (tiple): velocity changes of horizontal and vertical direction. e.g. (ax, ay)
    @param noise_probability (None or scale): To make the task more challenging, 
            with probability noise_probability at each time step the velocity increments 
            are both zero, independently of the intended increments. None for not adding noise

    @return new_state (tuple): state after taking action
    @return reward (scal): reward for this step
    @return terminal (string): state of episode, 'F' for cross the finish line, 
            'B' for back to start line, 'C' for continue
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
        new_state, reward, terminal = None, FINISH_REWARD, 'F'
    elif area_flag == 0: # back start line
        new_state, reward, terminal = get_state_in_start_line(), OUT_REWARD, 'B'
    else:
        new_state, reward, terminal = (new_vx, new_vy, x+new_vx, y+new_vy), -1, 'C'
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

def get_action_from_q_value(state, q_value):
    '''sample an action from q_value function, e.g. action = argmax_a(q_value(state, a))
    '''
    vx, vy, x, y = state
    this_state_value = q_value[vx][vy][x][y]
    indexes = np.where(q_value[vx][vy][x][y]==np.max(q_value[vx][vy][x][y]))
    choosed = np.random.choice(indexes[0].shape[0])
    action = (ACTIONS[indexes[0][choosed]], ACTIONS[indexes[1][choosed]])
    return action

def get_action_from_policy_distribution(state, policy_distribution):
    '''sample an action from policy distribution  
    '''
    vx, vy, x, y = state
    action_ = np.random.choice(np.arange(9), p=policy_distribution[vx][vy][x][y].flatten())
    action = (ACTIONS[action_ // 3], ACTIONS[action_ % 3])
    return action

def play(policy_func=None, policy_distribution=None, q_value=None, initial_state=None, render=False):
    '''provide one and only one of policy_func, policy_distribution and q_value,
    generate an episode from initial_state using given policy
    @param policy_func: a function given a state that return an action and probability of the action
    @param policy_distribution (numpy array): probability of action=(ax, ay) from state=(vx,vy,x,y) 
            is policy_distribution[vx][vy][x][y][ax][ay]
    @param q_value: value of action (ax, ay) from state (vx, vy, x, y) is q_value[vx][vy][x][y][ax][ay]
    @param initial_state: the initial state of the episode, if None, generate a random one 
    @param render: whether visualize the episodes or not
    @return trajectories: a list of tuples (state, action, reward)
    '''
    if initial_state is not None:
        current_state = initial_state
    else:
        current_state = get_state_in_start_line()
    trajectories = []
    while True:
        if policy_func is not None:
            action, _ = policy_func(current_state)
        elif policy_distribution is not None:
            action = get_action_from_policy_distribution(current_state, policy_distribution)
        elif q_value is not None:
            action = get_action_from_q_value(current_state, q_value)
        else:
            print('One and only one of policy_func, policy_distribution and q_value must be provided!')
            raise
        next_state, reward, terminal = take_action(current_state, action, noise_probability=0.1)
        trajectories.append((copy(current_state), action, reward, terminal))
        current_state = next_state
        if terminal == 'F':
            break
    if render:
        render_trajectory(trajectories) 
    return trajectories

def on_policy_mc_control(eposodes, gamma=1, epsilon=0.1, save_dir=None, render_every=None, save_every=None):
    '''off policy (every vist) MC control incremental implementation
    '''

    def initilize():
        '''initilize q_value and returns and policy distribution
        '''
        q_value = np.empty(shape=(6, 6, MAP_WIDTH, MAP_HEIGHT, 3, 3)) # q_value[vx][vy][x][y][ax][ay]
        q_value.fill(-float('inf'))
        for vx in range(6):
            for vy in range(6):
                for x in range(MAP_WIDTH):
                    for y in range(MAP_HEIGHT):
                        possibile_actions = get_legal_actions(state=(vx, vy, x, y))
                        for action in possibile_actions:
                            q_value[vx][vy][x][y][action[0]+1][action[1]+1] = 0
        returns = [[[[[[[] for ay in range(3)] \
                        for ax in range(3)] \
                        for y in range(MAP_HEIGHT)] \
                        for x in range(MAP_WIDTH)] \
                        for vy in range(6)] \
                        for vx in range(6)]
        policy =  np.zeros(shape=(6, 6, MAP_WIDTH, MAP_HEIGHT, 3, 3))
        for vx in range(6):
            for vy in range(6):
                for x in range(MAP_WIDTH):
                    for y in range(MAP_HEIGHT):
                        possibile_actions = get_legal_actions(state=(vx, vy, x, y))
                        prob = 1.0 / len(possibile_actions)
                        for action in possibile_actions:
                            policy[vx][vy][x][y][action[0]+1][action[1]+1] = prob
        return q_value, returns, policy

    q_value, returns, policy = initilize()

    def update_policy(epsilon, state, best_action, policy):
        '''update policy in-place
        '''
        valid_actions = get_legal_actions(state)
        vx, vy, x, y = state
        for action in valid_actions:
            if action == best_action:
                policy[vx][vy][x][y][action[0]+1][action[1]+1] = \
                                                    1 - epsilon + epsilon / len(valid_actions)
            else:
                policy[vx][vy][x][y][action[0]+1][action[1]+1] = epsilon / len(valid_actions)

    for i in tqdm(range(eposodes)):
        trajectories = play(policy_distribution=policy)
        if render_every is not None and (i+1) % render_every == 0:
            render_trajectory(trajectories)
        g = 0
        for t, traj in enumerate(reversed(trajectories)):
            g = gamma * g + traj[2]
            vx, vy, x, y = traj[0]
            ax, ay = traj[1]
            returns[vx][vy][x][y][ax+1][ay+1].append(g)
            q_value[vx][vy][x][y][ax+1][ay+1] = \
                    sum(returns[vx][vy][x][y][ax+1][ay+1]) / len(returns[vx][vy][x][y][ax+1][ay+1])
            best_action = get_action_from_q_value(traj[0], q_value)
            update_policy(epsilon, traj[0], best_action, policy)

        if save_every is not None:
            if (i+1) % save_every == 0:
                np.save(os.path.join(save_dir, 'q_value_'+str(i+1)+'.npy'), q_value)
                np.save(os.path.join(save_dir, 'policy_'+str(i+1)+'.npy'), policy)
    np.save(os.path.join(save_dir, 'q_value_'+str(i+1)+'.npy'), q_value)
    np.save(os.path.join(save_dir, 'policy_'+str(i+1)+'.npy'), policy)
    return q_value, policy

def off_policy_mc_control(episodes, gamma=1, save_dir=None, render_every=None, save_every=None):
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
                            q_value[vx][vy][x][y][action[0]+1][action[1]+1] = 0
        cumulative_weights = np.zeros(shape=q_value.shape, dtype=np.int32)
        target_policy = [[[[get_action_from_q_value(state=(vx, vy, x, y), q_value=q_value) 
                            for y in range(MAP_HEIGHT)] \
                            for x in range(MAP_WIDTH)] \
                            for vy in range(6)] \
                            for vx in range(6)] 

        return q_value, cumulative_weights, target_policy

    q_value, cumulative_weights, target_policy = initilize()
    for i in tqdm(range(episodes)):
        trajectories = play(policy_func=random_policy)
        g = 0
        w = 1
        for t, traj in enumerate(reversed(trajectories)):
            g = gamma * g + traj[2]
            action = traj[1]
            vx, vy, x, y = traj[0]
            cumulative_weights[vx][vy][x][y][action[0]+1][action[1]+1] += w
            q_value[vx][vy][x][y][action[0]+1][action[1]+1] += \
                    w / cumulative_weights[vx][vy][x][y][action[0]+1][action[1]+1] * \
                    (g - q_value[vx][vy][x][y][action[0]+1][action[1]+1])
            target_policy[vx][vy][x][y] = get_action_from_q_value(state=(vx, vy, x, y), q_value=q_value)
            
            if action != target_policy[vx][vy][x][y]:
                break
            _, behavior_action_probability = random_policy(state=(vx, vy, x, y))
            w /= behavior_action_probability
        if save_every is not None and (i+1) % save_every == 0:
            np.save(os.path.join(save_dir, 'q_value_'+str(i+1)+'.npy'), q_value)
            np.save(os.path.join(save_dir, 'policy_'+str(i+1)+'.npy'), target_policy)

        if render_every is not None and (i+1) % render_every == 0:
            play(q_value=q_value, render=True)

    np.save(os.path.join(save_dir, 'q_value_'+str(i+1)+'.npy'), q_value)
    np.save(os.path.join(save_dir, 'policy_'+str(i+1)+'.npy'), target_policy)
    return q_value, target_policy

def render_trajectory(trajectories, speed_ratio=1):
    turtle.clear()
    turtle.pencolor(np.random.rand(), np.random.rand(), np.random.rand())
    initial_state = trajectories[0][0]
    turtle.penup()
    turtle.setpos(COORDINATE_ORIGIN[0]+initial_state[2]*COORDINATE_UNIT,
            COORDINATE_ORIGIN[1] + initial_state[3]*COORDINATE_UNIT)
    turtle.pendown()
    turtle.dot(size=5)
    i = 0
    while i < len(trajectories):
        traj = trajectories[i]
        vx, vy, x, y = traj[0]

        vx *= COORDINATE_UNIT
        vy *= COORDINATE_UNIT

        turtle.speed(vx*speed_ratio)
        turtle.forward(vx)

        turtle.left(90)
        turtle.speed(vy*speed_ratio)
        turtle.forward(vy)
        turtle.dot(size=5)
        turtle.right(90)

        if traj[-1] == 'B':
            i += 1
            turtle.clear()
            initial_state = trajectories[i][0]
            assert initial_state[0] == 0 and initial_state[1] == 0
            turtle.penup()
            turtle.setpos(COORDINATE_ORIGIN[0]+initial_state[2]*COORDINATE_UNIT,
                    COORDINATE_ORIGIN[1] + initial_state[3]*COORDINATE_UNIT)
            turtle.pendown()
            turtle.dot(size=5)
            continue
        i += 1

if __name__ == '__main__':
    # off policy 
    # train 
    # now_str = str(datetime.now())[:-7].replace('-', '_').replace(':', '_').replace(' ', '_') + '_off_policy'
    # model_dir = os.path.join('racetrack_result', now_str)
    # os.mkdir(model_dir)
    # off_policy_mc_control(episodes=10000, gamma=1, save_dir=model_dir, render_every=1000, save_every=5000)
    
    # play
    # q_value = np.load(os.path.join('racetrack_result', '2019_03_02_23_47_11_off_policy', 'q_value_100000.npy'))
    # while True:
    #     play(q_value=q_value, render=True)
    
    # on policy

    # learing
    # now_str = str(datetime.now())[:-7].replace('-', '_').replace(':', '_').replace(' ', '_')
    # model_dir = os.path.join('racetrack_result', now_str)
    # os.mkdir(model_dir)
    # on_policy_mc_control(eposodes=100000, 
    #                     gamma=0.9, 
    #                     epsilon=0.1, 
    #                     save_dir=model_dir, 
    #                     render_every=1000, 
    #                     save_every=1000)

    # play
    q_value = np.load(os.path.join('racetrack_result', '2019_03_03_14_55_55', 'q_value_24000.npy'))
    # while True:
    play(q_value=q_value, render=True)
    time.sleep(1000)