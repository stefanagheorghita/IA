import numpy as np

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


def initialize_table():
    Q = {}
    rows = 7
    cols = 10
    for i in range(rows):
        for j in range(cols):
            state = (i, j)
            for action in ['up', 'down', 'left', 'right']:
                Q[state, action] = 0
    return Q


def initialize_parameters():
    learning_rate = 0.1
    discount_factor = 0.5
    epsilon = 1
    episodes = 1000
    return learning_rate, discount_factor, epsilon, episodes


def get_next_state(state, action):
    new_row = state[0]
    new_col = state[1]
    if action == 'up':
        new_row = state[0] - 1 - wind[state[1]]
    elif action == 'down':
        new_row = state[0] + 1 - wind[state[1]]
    elif action == 'left':
        new_col = state[1] - 1
        new_row = state[0] - wind[state[1]]
    elif action == 'right':
        new_col = state[1] + 1
        new_row = state[0] - wind[state[1]]
    if new_row < 0:
        new_row = 0
    if new_row > 6:
        new_row = 6
    if new_col < 0:
        new_col = 0
    if new_col > 9:
        new_col = 9
    next_state = (new_row, new_col)
    return next_state


def get_max_value(Q, state):
    value = -1000
    actions = ["up", "down", "left", "right"]
    for action in actions:
        if Q[state, action] > value:
            value = Q[state, action]
    return value


def get_max_action(Q, state):
    value = -1000
    actions = ["up", "down", "left", "right"]
    max_action = []
    for action in actions:
        if Q[state, action] > value:
            value = Q[state, action]
            max_action = [action]
        elif Q[state, action] == value:
            max_action.append(action)
    rand = np.random.randint(0, len(max_action))
    return max_action[rand]


def algorithm(initial_state):
    Q = initialize_table()
    learning_rate, discount_factor, epsilon, episodes = initialize_parameters()
    # row = 0
    # col = 0
    for episode in range(episodes):
        # if row == 6 and col == 9:
        #     row = 0
        #     col = 0
        # if col == 9:
        #     col = 0
        #     row += 1
        # else:
        #     col += 1
        # row = np.random.randint(0, 7)
        # col = np.random.randint(0, 10)
        # state = (row, col)
        state = initial_state
        num_steps = 0
        while state != (3, 7) and num_steps < 500:
            num_steps += 1
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(['up', 'down', 'left', 'right'])
            else:
                action = get_max_action(Q, state)
            next_state = get_next_state(state, action)
            if next_state == (3, 7):
                reward = 100
            else:
                reward = -1
            Q[state, action] += learning_rate * (
                    reward + discount_factor * get_max_value(Q, next_state) - Q[state, action])
            state = next_state
            epsilon = epsilon * 0.999

    return Q


def get_policy(Q):
    policy = {}
    for i in range(7):
        for j in range(10):
            state = (i, j)
            policy[state] = get_max_action(Q, state)

    return policy


def print_policy(policy):
    for i in range(7):
        for j in range(10):
            state = (i, j)
            print("For state ", state, " the best action is ", policy[state])
        print()
    return


def follow_path(policy, initial_state):
    state = initial_state
    num_steps = 0
    while state != (3, 7) and num_steps < 1000:
        num_steps += 1
        print("-" * 50)
        print("We are at step: ", num_steps)
        print("\033[34mState: ", state, "\033[0m")
        action = policy[state]
        print("\033[35mAction: ", action, "\033[0m")
        state = get_next_state(state, action)
    print(state)


if __name__ == '__main__':
    initial_state = (3, 0)
    Q = algorithm(initial_state)
    policy = get_policy(Q)
    # print_policy(policy)
    for i in range(7):
        for j in range(10):
            print("For state ", (i, j), " the best action is ", policy[(i, j)])
            print("up", Q[(i, j), 'up'], "down", Q[(i, j), 'down'], "left", Q[(i, j), 'left'], "right",
                  Q[(i, j), 'right'])
    follow_path(policy, initial_state)
