import QuickMaths as qm
import random
import os
import time
import copy
import quickbrain as qb

#A SMALL Q-LEARNING EXPERIMENT BUILT ON QUICKBRAIN
#RUNS IN THE CONSOLE, BASICALLY A MAZE SOLVER

MAP_SIZE = (10, 10)
DISCOUNT = 0.5
STATE_SPACE = MAP_SIZE[0] * MAP_SIZE[1]
ACTION_SPACE = 4
POSITIVE = 10
NEGATIVE = -1000
LEARNING_RATE = 1

direction_names = {
    0 : 'down',
    1 : 'up',
    2 : 'right',
    3 : 'left',
}

def space(x):
    if x == 0:
        x = " "
    return x

class Agent:
    def __init__(self):
        self.pos = [0,0]
        self.ch = 'X'
        self.brain = qb.Custom_DFF([4, 5, 4], qm.Activations.sigmoid)
        self.decision = 0

    def move(self, direction):
        if direction == 0 and self.pos[0] < MAP_SIZE[0] - 1:
            self.pos[0] += 1
        elif direction == 1 and self.pos[0] > 0:
            self.pos[0] -= 1
        elif direction == 2 and self.pos[1] < MAP_SIZE[1] - 1:
            self.pos[1] += 1
        elif direction == 3 and self.pos[1] > 0:
            self.pos[1] -= 1
        else:
            self.pos = self.pos
        self.decision = direction

        return direction
    def random_move(self):
        decision = random.randint(0, 3)
        self.move(decision)
        self.decision = decision
        return decision

class Food:
    def __init__(self):
        self.pos = [random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])]
        self.ch = 'O'

class Food_fixed():
    def __init__(self):
        self.pos = [MAP_SIZE[0] - 1, MAP_SIZE[1] - 1]
        self.ch = 'O'

class Obstacle:
    def __init__(self, pos_x, pos_y):
        self.pos = [pos_x, pos_y]
        self.ch = '■'

def get_reward(agent, food, obstacles):
    for obs in obstacles:
        if agent.pos[0] == obs.pos[0] and agent.pos[1] == obs.pos[1]:
            return NEGATIVE
    if agent.pos[0] == food.pos[0] and agent.pos[1] == food.pos[1]:
        return POSITIVE
    if agent.pos[0] <= 0:
        if agent.decision == 1:
           return NEGATIVE
    if agent.pos[0] >= MAP_SIZE[0] - 1:
        if agent.decision == 0:
            return NEGATIVE
    if agent.pos[1]  >= MAP_SIZE[1] - 1:
        if agent.decision == 2:
            return NEGATIVE
    if agent.pos[1] <= 0:
        if agent.decision == 3:
            return NEGATIVE
    return 0

def generate_obstacles(coords):
    r = []
    for coord in coords:
        r.append(Obstacle(coord[0], coord[1]))
    return r

def render(agent, env, food, obstacles):
    img = qm.clear_matrix(MAP_SIZE[0], MAP_SIZE[1])
    img.applyFunc(space)
    img[food.pos[0]][food.pos[1]] = food.ch
    for obs in obstacles:
        img[obs.pos[0]][obs.pos[1]] = obs.ch
    img[agent.pos[0]][agent.pos[1]] = agent.ch
    print(img)
    print(agent.decision)
    time.sleep(0.1)
    os.system("clear")

#An abandoned attempt at using Deep Q-learning for this task
'''
def DEEP_Q():
    obstacles = generate_obstacles([[5,5]])
    Q = qm.clear_matrix(MAP_SIZE[0], MAP_SIZE[1])
    agent = Agent()
    food = Food()
    print(food.pos)
    rand = True
    while True:
        print("random choices: ", rand)
        state = qm.Matrix(data=[[agent.pos[0], agent.pos[1], food.pos[0], food.pos[1]]]).transpose()
        thought = agent.brain.feed_forward(state)
        agent.decision= thought.get_max_index()[0]
        if rand:
            action = agent.random_move()
        else:
            agent.move(agent.decision)
            action = agent.decision
        R = get_reward(agent, food, obstacles)
        print("Reward: ", R)
        print("Agent decision: ", agent.decision)
        new_state = qm.Matrix(data=[[agent.pos[0], agent.pos[1], food.pos[0], food.pos[1]]]).transpose()
        new_decision = agent.brain.feed_forward(state).get_max_index()[0]

        Q_target = qm.Matrix(4, 1)
        Q_target.clear()
        Q_target[action][0] = R + DISCOUNT * new_decision

        E = Q_target - action
        print("Target Q: ", Q_target)
        print("Thought: ", thought)
        print("Direction: ", direction_names[agent.decision])

        agent.brain.backpropagate(state, E, LEARNING_RATE)

        try:
            render(agent, Q, food, obstacles)
        except IndexError:
            print(agent.pos)
            agent.pos = [0, 0]
            food = Food()
        except KeyboardInterrupt:
            rand = not rand
        if agent.pos[0] == food.pos[0] and agent.pos[1] == food.pos[1]:
            agent.pos = [0, 0]
            food = Food()
'''

def NORMAL_Q():
    obstacles = generate_obstacles([
        [1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],
        [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],
        [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],[5,8],
        [7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],

        ]
        )
    Q = qm.clear_matrix(MAP_SIZE[0], MAP_SIZE[1])
    agent = Agent()
    food = Food_fixed()

    def obs_2_coords(obs):
        r = [0,0]
        r[1] = obs % 10
        r[0] = int((obs - r[1]) / MAP_SIZE[1])

        return r
    def get_exp():
        try:
            observations = []
            table = qm.random_matrix(STATE_SPACE, ACTION_SPACE)
            while True:
                action = agent.random_move()
                obs = (agent.pos[0]) * 10 + agent.pos[1]
                R = get_reward(agent, food)

                action2 = agent.random_move()
                obs2 = (agent.pos[0]) * 10 + agent.pos[1]
                R2 = get_reward(agent, food)

                table[obs][action] = R + DISCOUNT * max(table[obs2])
                print(table)
                print(table.rows)
                print(obs)
                observations.append(obs2)
                render(agent, Q, food)

                if agent.pos[0] == food.pos[0] and agent.pos[1] == food.pos[1]:
                    agent.pos = [0, 0]


        except KeyboardInterrupt:
            return table
    table = qm.random_matrix(STATE_SPACE, ACTION_SPACE)
    rand = True
    while True:
        try:
                obs = (agent.pos[0]) * 10 + agent.pos[1]
                possible_actions = table[obs]
                max_index = possible_actions.index(max(possible_actions))
                if not rand:
                    print(max_index)
                    action = max_index
                    print(possible_actions[max_index])
                else:
                    action = agent.random_move()
                agent.move(action)
                R = get_reward(agent, food, obstacles)
                obs2 = (agent.pos[0]) * 10 + agent.pos[1]

                table[int(obs)][int(action)] = R + DISCOUNT * max(table[int(obs2)])
                #print(table)
                if not rand:
                    print("Smart mode")
                else:
                    print("Stoopid")
                #print(R)
                render(agent, Q, food, obstacles)

                if agent.pos[0] == food.pos[0] and agent.pos[1] == food.pos[1]:
                    agent.pos = [0, 0]
        except KeyboardInterrupt:
            rand = not rand



NORMAL_Q()