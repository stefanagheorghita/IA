import random
import math
import time

# Reprezentarea unei stari pentru problema data:
# [t0, t1, t2,..., tn] avand loc urmatoarele doua:
# 1. ti apartine {0, 1, 2,..., n-1}, pentru orice i din {1, 2,...,n}
# 2. not(ti = tj), oricare ar fi i, j din {1, 2,...,n}
# 3. t0 apartine {-1, 1, 2,...,n-1}

# Reprezentarea starii initiale pentru problema data:
# [t0, t1, t2,..., tn] avand loc urmatoarele doua:
# 1. ti apartine {0, 1, 2,..., n-1}, pentru orice i din {1, 2,...,n}
# 2. not(ti = tj), oricare ar fi i, j din {1, 2,...,n}
# 3. t0 = -1

# Reprezentarea starii finale pentru problema data:
# [t0, t1, t2,..., tn] avand loc urmatoarele:
# 1. ti apartine {0, 1, 2,..., n}, i apartine {1, 2,...,n}
# 2. not(ti = tj), oricare ar fi i, j din {1, 2,...,n}
# 3. t1 < t2 < ... < ti < ... < tn, oricare ar fi i din {1, 2,..., n}, unde not(ti==0) (pentru ti==0, verificam t(i-1)<t(i+1))
# 4. t0 apartine {-1,1, 2,...,n}

# Pentru problema in cauza avem n = 3*3
# Exemple:
[-1, 8, 0, 6, 5, 4, 7, 2, 3, 1]  # exemplu stare initiala
[5, 8, 4, 6, 5, 0, 7, 2, 3, 1]  # exemplu stare intermediara
[1, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # exemplu stare finala


def initial_state(n):
    # n e numarul de elemente ale vectorului
    # state = [n]
    if math.sqrt(n) != int(math.sqrt(n)):
        return []
    state = [-1] + list(range(n))
    shuffled_state = [state[0]] + random.sample(state[1:], len(state) - 1)
    return shuffled_state


def state_exists(state) -> bool:
    if len(state) != 10:
        return False
    if len(state[1:]) != len(set(state[1:])):
        return False
    for i in range(1, len(state) - 1):
        if not (state[i] in range(0, len(state) - 1)):
            return False
    if state[0] != -1:
        zero_pos = state.index(0)
        pos = state[1:].index(state[0]) + 1
        if zero_pos % 3 != pos % 3 and (zero_pos - 1) // 3 != (pos - 1) // 3:
            return False
        else:
            if (zero_pos - 1) // 3 == (pos - 1) // 3 and abs(pos - zero_pos) > 1:
                return False
            else:
                if zero_pos % 3 == pos % 3 and abs(pos - zero_pos) > 3:
                    return False
    return True


def is_final_state(state) -> bool:
    if not (state_exists(state)):
        print("The state is invalid.")
        return False
    for i in range(1, len(state) - 1):
        if not (state[i] == 0) and not (state[i + 1] == 0):
            if state[i] != state[i + 1] - 1:
                return False
        elif state[i] == 0 and i != 1 and state[i - 1] != state[i + 1] - 1:
            return False
    return True


def valid_transition(state, first_value, second_value) -> bool:
    if not (state_exists(state)):
        print("The state is invalid.")
        return False
    if first_value != 0 and second_value != 0:
        return False
    elif first_value == 0 and second_value == 0:
        return False
    elif first_value == second_value:
        return False
    elif not (first_value in range(0, len(state) - 1)) or not (second_value in range(0, len(state) - 1)):
        return False
    else:
        if first_value != 0:
            if state[0] == first_value:
                return False
        elif second_value != 0:
            if state[0] == second_value:
                return False
        if first_value == 0:
            value_zero = first_value
            value = second_value
        else:
            value_zero = second_value
            value = first_value
        position_zero = state.index(value_zero)
        position = state.index(value)
        if position_zero % 3 == position % 3 and (position_zero - 1) // 3 == (position - 1) // 3:
            return True
        elif (position_zero - 1) // 3 == (position - 1) // 3 and abs(position - position_zero) == 1:
            return True
        elif position_zero % 3 == position % 3 and abs(position - position_zero) == 3:
            return True
    return False


def transition(state, first_value, second_value):
    if not valid_transition(state, first_value, second_value):
        print("\033[33m-- state=" + str(state) + ":\033[0m The transition " + str(first_value) + "-" + str(
            second_value) + " is invalid! We remain in the same state.")
        return state
    else:
        if first_value == 0:
            value = second_value
        else:
            value = first_value
        print("\033[93m-- state=" + str(state) + ":\033[0m The transition " + str(first_value) + "-" + str(
            second_value) + " is valid!")
        pos1 = state.index(first_value)
        pos2 = state.index(second_value)
        new_state = state.copy()
        new_state[pos1], new_state[pos2] = state[pos2], state[pos1]
        new_state[0] = value

        return new_state


visited = set()


def search(state, depth, max_depth):
    global visited
    if depth > max_depth:
        return False
    if is_final_state(state):
        print("\033[92mSOLUTION: " + str(state) + "\033[0m")
        return state
    print("\033[95mCurrent state: " + str(state) + "\033[0m")
    if depth != max_depth:
        for value in range(1, 9):
            new_state = transition(state, value, 0)
            if new_state == state:
                continue
            rez = search(new_state, depth + 1, max_depth)
            if rez:
                return rez
        print("\033[36m state=" + str(
            state) + " We visited all neighbours for this state. We return to parent state.\033[0m\n ")
    else:
        print("\033[94mstate=" + str(
            state) + " We reached the last level of this depth. We return to parent state.\033[0m\n")
    return False


def alg_IDDFS(state):
    for depth in range(1, 30):
        print("\n\n\033[91m We are at depth " + str(depth) + "\033[0m")
        res = search(state, 0, depth)
        print("res" + str(res))
        if res:
            return res, depth
    return False, None


##### TEMA 2 #####


final_states = [
    [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    [-1, 1, 0, 2, 3, 4, 5, 6, 7, 8],
    [-1, 1, 2, 0, 3, 4, 5, 6, 7, 8],
    [-1, 1, 2, 3, 0, 4, 5, 6, 7, 8],
    [-1, 1, 2, 3, 4, 0, 5, 6, 7, 8],
    [-1, 1, 2, 3, 4, 5, 0, 6, 7, 8],
    [-1, 1, 2, 3, 4, 5, 6, 0, 7, 8],
    [-1, 1, 2, 3, 4, 5, 6, 7, 0, 8],
    [-1, 1, 2, 3, 4, 5, 6, 7, 8, 0],
]


def manhattan_distance(state):
    distance = 0
    for i in range(1, len(state)):
        if state[i] != 0:
            current_row = (i - 1) // 3
            current_col = (i - 1) % 3
            final_row_w0 = (state[i] - 1) // 3
            final_col_w0 = (state[i] - 1) % 3
            final_row_w1 = state[i] // 3
            final_col_w1 = state[i] % 3
            # distance = distance + min(abs(current_row - final_row_w0),abs(current_row - final_row_w1)) + min(abs(current_col - final_col_w0),abs(current_col - final_col_w1))
            mini = min(abs(current_row - final_row_w0) + abs(current_col - final_col_w0),
                       abs(current_row - final_row_w1) + abs(current_col - final_col_w1))
            distance = distance + mini
    return distance


def manhattan_distance_2(state):
    distances = []
    for final_state in final_states:
        distance = 0
        for i in range(1, len(state)):
            if state[i] != 0:
                current_row = (i - 1) // 3
                current_col = (i - 1) % 3
                final_row = (final_state.index(state[i]) - 1) // 3
                final_col = (final_state.index(state[i]) - 1) % 3
                mini = abs(current_row - final_row) + abs(current_col - final_col)
                distance = distance + mini
        distances.append(distance)
    return min(distances)


def hamming_distance(state):
    distance = 0
    for i in range(1, len(state)):
        if state[i] != 0:
            final_pos_w0 = state[i]
            final_pos_w1 = state[i] + 1
            if i != final_pos_w1 and i != final_pos_w0:
                distance += 1
    return distance


def hamming_distance_2(state):
    distances = []
    for final_state in final_states:
        dist = 0
        for i in range(1, len(state)):
            if state[i] != 0 and state[i] != final_state[i]:
                dist += 1
        distances.append(dist)
    min_distance = min(distances)
    return min_distance


def wrong_row_col(state):
    distance = 0
    for i in range(1, len(state)):
        if state[i] != 0:
            current_row = (i - 1) // 3
            current_col = (i - 1) % 3
            final_row_w0 = (state[i] - 1) // 3
            final_col_w0 = (state[i] - 1) % 3
            final_row_w1 = state[i] // 3
            final_col_w1 = state[i] % 3
            if current_row != final_row_w0 and current_row != final_row_w1:
                distance += 1
            if current_col != final_col_w0 and current_col != final_col_w1:
                distance += 1
    return distance


def euclidian_distance(state):
    distance = 0
    for i in range(1, len(state)):
        if state[i] != 0:
            current_row = (i - 1) // 3
            current_col = (i - 1) % 3
            final_row_w0 = (state[i] - 1) // 3
            final_col_w0 = (state[i] - 1) % 3
            final_row_w1 = state[i] // 3
            final_col_w1 = state[i] % 3
            mini = min(math.sqrt((current_row - final_row_w0) ** 2 + (current_col - final_col_w0) ** 2),
                       math.sqrt((current_row - final_row_w1) ** 2 + (current_col - final_col_w1) ** 2))
            distance = distance + mini
    return distance


def find_mismatch(state, final_state):
    for i in range(len(state)):
        if state[i] != 0 and state[i] != final_state[i]:
            return i


def relaxing(init_state):
    distances = []
    init_state = init_state[1:]
    for final_state in final_states:
        state = init_state.copy()
        dist = 0
        final_state = final_state[1:]
        while state != final_state:
            zeroPos1 = state.index(0)
            zeroPos2 = final_state.index(0)
            if zeroPos1 == zeroPos2:
                index = find_mismatch(state, final_state)
                x = state[zeroPos1]
                y = state[index]
                state[index] = x
                state[zeroPos1] = y
                dist += 1
            else:
                x = state[zeroPos1]
                y_index = state.index(final_state[zeroPos1])
                y = state[y_index]
                state[zeroPos1] = y
                state[y_index] = x
                dist += 1
        if dist == 0:
            return dist
        distances.append(dist)
    return min(distances)


def greedy(state, h):
    moves = -1
    next_states = []
    if h == 1:
        next_states = [(state, manhattan_distance(state), 0)]
    elif h == 2:
        next_states = [(state, manhattan_distance_2(state), 0)]
    elif h == 3:
        next_states = [(state, hamming_distance(state), 0)]
    elif h == 4:
        next_states = [(state, hamming_distance_2(state), 0)]
    elif h == 5:
        next_states = [(state, wrong_row_col(state), 0)]
    elif h == 6:
        next_states = [(state, euclidian_distance(state), 0)]
    elif h == 7:
        next_states = [(state, relaxing(state), 0)]
    else:
        print("Wrong parameter")
        return
    explored = set()
    while next_states:
        next_states.sort(key=lambda x: x[1])
        moves += 1
        current_state, dist, depth = next_states[0]
        next_states = next_states[1:]
        print("We are at state: " + str(current_state))
        if is_final_state(current_state):
            return current_state, depth, moves
        explored.add(tuple(current_state))
        for i in range(1, 9):
            next_state = transition(current_state, i, 0)
            if next_state != current_state:
                if tuple(next_state) not in explored:
                    if next_state not in [state1 for state1, _, _ in next_states]:
                        if h == 1:
                            next_states.append((next_state, manhattan_distance(next_state), depth + 1))
                        elif h == 2:
                            next_states.append((next_state, manhattan_distance_2(next_state), depth + 1))
                        elif h == 3:
                            next_states.append((next_state, hamming_distance(next_state), depth + 1))
                        elif h == 4:
                            next_states.append((next_state, hamming_distance_2(next_state), depth + 1))
                        elif h == 5:
                            next_states.append((next_state, wrong_row_col(next_state), depth + 1))
                        elif h == 6:
                            next_states.append((next_state, euclidian_distance(next_state), depth + 1))
                        elif h == 7:
                            next_states.append((next_state, relaxing(next_state), depth + 1))

    return None, -1, -1


def a_star(state):
    priorities = [(state, 0)]
    costs = {tuple(state): 0}
    while priorities:
        current_state, total_cost = priorities[0]
        priorities = priorities[1:]
        if is_final_state(current_state):
            return current_state, total_cost

        for i in range(1, 9):
            next_state = transition(current_state, i, 0)
            if next_state != current_state:
                new_cost = costs[tuple(current_state)] + 1
                if tuple(next_state) not in costs or new_cost < costs[tuple(next_state)]:
                    costs[tuple(next_state)] = new_cost
                    priority = new_cost + manhattan_distance(next_state)
                    priorities.append((next_state, priority))
                    priorities.sort(key=lambda x: x[1])
    return None, -1


def solving(states):
    dictio = {
        0: "IDDFS:",
        1: "Greedy with Manhattan distance:",
        2: "Greedy with Manhattan distance 2:",
        3: "Greedy with Hamming distance:",
        4: "Greedy with Hamming distance 2:",
        5: "Greedy with number of pieces in the wrong row and col:",
        6: "Greedy with euclidian distance:",
        7: "Greedy with relaxing moves:",
        8: "A*"
    }
    results = {}
    for state in states:
        for h in range(0, 9):
            if h == 0:
                start_time = time.time()
                results[str(state) + 'result_' + str(h)], results[str(state) + 'depth_' + str(h)] = alg_IDDFS(state)
                end_time = time.time()
                results[str(state) + 'exec_time_' + str(h)] = end_time - start_time
            elif h == 8:
                start_time = time.time()
                results[str(state) + 'result_' + str(h)], results[str(state) + 'depth_' + str(h)] = a_star(state)
                end_time = time.time()
                results[str(state) + 'exec_time_' + str(h)] = end_time - start_time
            else:
                start_time = time.time()
                results[str(state) + 'result_' + str(h)], results[str(state) + 'depth_' + str(h)], results[
                    str(state) + 'moves_' + str(h)] = greedy(state, h)
                end_time = time.time()
                results[str(state) + 'exec_time_' + str(h)] = end_time - start_time
    for state in states:
        print("\n\n\033[1;1m Instance \033[0m" + str(state))
        for h in range(0, 9):
            if h == 0 or h == 8:
                print("\033[105m" + dictio[h] + "\033[0m")
                print(" Result: " + str(results[str(state) + 'result_' + str(h)]))
                print(" Depth: " + str(results[str(state) + 'depth_' + str(h)]))
                print(" Execution time: " + str(results[str(state) + 'exec_time_' + str(h)]))
            else:
                print("\033[105m" + dictio[h] + "\033[0m")
                print(" Result: " + str(results[str(state) + 'result_' + str(h)]))
                print(" Depth: " + str(results[str(state) + 'depth_' + str(h)]))
                print(" Moves: " + str(results[str(state) + 'moves_' + str(h)]))
                print(" Execution time: " + str(results[str(state) + 'exec_time_' + str(h)]))


instances = [
    [-1, 8, 6, 7, 2, 5, 4, 0, 3, 1],
    [-1, 2, 5, 3, 1, 0, 6, 4, 7, 8],
    [-1, 2, 7, 5, 0, 8, 4, 3, 1, 6]
]

solving(instances)
# print(a_star([-1, 8, 6, 7, 2, 5, 4, 0, 3, 1]))
