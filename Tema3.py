# Reprezentarea unei stari:
#   Stare initiala: vector de 9 elemente, ce contine cifrele de la 1 la 9 in ordine crescatoare
#                   pe pozitia i se va afla valoarea i+1: v[i] = i+1
#   Stare intermediara: la alegerea unui numar x de catre un jucator, se va modifica pozitia x-1,
#                       astfel incat sa fie inlocuita cu simbolul jucatorului care a ales numarul (A sau B)

magic_square = [
    [8, 1, 6],
    [3, 5, 7],
    [4, 9, 2],
]


def initialize():
    return [8, 1, 6, 3, 5, 7, 4, 9, 2]


def is_final_state(state):
    # values_A = [i for i in range(9) if state[i] == 'A']
    # values_B = [i for i in range(9) if state[i] == 'B']
    # for first in range(len(values_A)):
    #     for second in range(first + 1, len(values_A)):
    #         for third in range(second + 1, len(values_A)):
    #             if values_A[first] + values_A[second] + values_A[third] == 15:
    #                 return True, 'A'
    #
    # for first in range(len(values_B)):
    #     for second in range(first + 1, len(values_B)):
    #         for third in range(second + 1, len(values_B)):
    #             if values_B[first] + values_B[second] + values_B[third] == 15:
    #                 return True, 'B'
    for i in range(3):
        if state[i * 3: i * 3 + 3].count('A') == 3 or state[i::3].count('A') == 3:
            return True, 'A'
        if state[i * 3: i * 3 + 3].count('B') == 3 or state[i::3].count('B') == 3:
            return True, 'B'
    if state[0:9:4].count('A') == 3 or state[2:7:2].count('A') == 3:
        return True, 'A'
    if state[0:9:4].count('B') == 3 or state[2:7:2].count('B') == 3:
        return True, 'B'
    k = 0
    for i in range(9):
        if not isinstance(state[i], int):
            k += 1
    if k == len(state):
        return True, 'remiza'
    return False, None


def validate_transition(state, number):
    for i in range(len(state)):
        if state[i] == number:
            return True
    return False


def transition(state, number, player):
    if validate_transition(state, number) is False:
        return None
    pos = state.index(number)
    new_state = state.copy()
    new_state[pos] = player
    # is_final, winner = is_final_state(new_state)
    # if is_final is True:
    #     return new_state, winner
    return new_state


def heuristic(state):
    value = 0
    value_row = 0
    value_col = 0
    value_diag1 = 0
    value_diag2 = 0
    if is_final_state(state)[0]:
        if is_final_state(state)[1] == 'A':
            return 100
        elif is_final_state(state)[1] == 'B':
            return -100
        else:
            return 0
    for i in range(3):
        if state[i * 3: i * 3 + 3].count('A') == 2 and state[i * 3: i * 3 + 3].count('B') == 0:
            value_row += 2
        if state[i * 3: i * 3 + 3].count('A') == 1 and state[i * 3: i * 3 + 3].count('B') == 0:
            value_row += 1
        if state[i * 3: i * 3 + 3].count('B') == 2 and state[i * 3: i * 3 + 3].count('A') == 0:
            value_row -= 2
        if state[i * 3: i * 3 + 3].count('B') == 1 and state[i * 3: i * 3 + 3].count('A') == 0:
            value_row -= 1
        if state[i::3].count('A') == 2 and state[i::3].count('B') == 0:
            value_col += 2
        if state[i::3].count('A') == 1 and state[i::3].count('B') == 0:
            value_col += 1
        if state[i::3].count('B') == 2 and state[i::3].count('A') == 0:
            value_col -= 2
        if state[i::3].count('B') == 1 and state[i::3].count('A') == 0:
            value_col -= 1
    if state[0:9:4].count('A') == 2 and state[0:9:4].count('B') == 0:
        value_diag1 += 2
    if state[0:9:4].count('A') == 1 and state[0:9:4].count('B') == 0:
        value_diag1 += 1
    if state[0:9:4].count('B') == 2 and state[0:9:4].count('A') == 0:
        value_diag1 -= 2
    if state[0:9:4].count('B') == 1 and state[0:9:4].count('A') == 0:
        value_diag1 -= 1
    if state[2:7:2].count('A') == 2 and state[2:7:2].count('B') == 0:
        value_diag2 += 2
    if state[2:7:2].count('A') == 1 and state[2:7:2].count('B') == 0:
        value_diag2 += 1
    if state[2:7:2].count('B') == 2 and state[2:7:2].count('A') == 0:
        value_diag2 -= 2
    value = value_row + value_col + value_diag1 + value_diag2
    return value


def generate_states(state, player):
    states = []
    for i in range(9):
        new_state = transition(state, i + 1, player)
        if new_state is not None:
            states.append(new_state)
    return states


def minimax(depth, state, player):
    if is_final_state(state)[0] or depth == 0:
        #  print(state, is_final_state(state), heuristic(state, player), depth)
        return state, heuristic(state)
    if player == 'B':
        states = generate_states(state, 'B')
        best = float('inf')
        for st in states:
            _, val = minimax(depth - 1, st, 'A')
            if val < best:
                best = val
                good_state = st
        return good_state, best

    if player == 'A':
        states = generate_states(state, 'A')
        best = float('-inf')

        for st in states:
            _, val = minimax(depth - 1, st, 'B')
            if val > best:
                best = val
                good_state = st
        return good_state, best


def calculate():
    state = initialize()
    current_player = 'A'
    print(state)
    while not is_final_state(state)[0]:
        if current_player == 'A':
            number = int(input("Choose a number: "))
            new_state = transition(state, number, current_player)
            print("PLayer A: ", new_state)
            if new_state is None:
                print("Invalid number!")
                continue
            state = new_state
            current_player = 'B'
        else:
            new_state, best = minimax(3, state, 'B')
            current_player = 'A'
            state = new_state
            print("Player B: ", state)
    print(is_final_state(state))


calculate()
