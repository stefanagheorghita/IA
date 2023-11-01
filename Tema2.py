# Multimea de variabile: Xij, i din {0,...8}, j din {0,...8}, unde Xij reprezinta valoarea din celula i,j;
#                        vom reprezenta sudoku-ul ca o matrice de 9x9: sudoku[i][j] = Xij
# Domeniul variabilelor:
#               {1,...,9}-pentru celulele completate, pentru majoritatea variabilelor Xij s
#               i {2, 4, 6, 8} pentru anumite valori ale lui i si j - pentru starea finala
#               {-1}-pentru celulele necompletate ce pot avea orice valoare
#               {0}-pentru celulele ce pot lua numai valori pare
#               Domeniu complet: {-1, 0, 1,..., 9}
# Restrictii: C = {C1, C2, C3, C4}
#             C1: Xij != Xik, i din {0,...8}, j din {0,...8}, k din {0,...8}, j != k
#             C2: Xij != Xkj, i din {0,...8}, j din {0,...8}, k din {0,...8}, i != k
#             C3: Xij != Xxy, i din {0,...8}, j din {0,...8}, x din {0,...8}, y din {0,...8}, (i,j) != (x,y)
#               oricare ar fi i, j, x, y, cu proprietatea ca Xij, Xxy apartin aceluiasi Sab,
#               cu a, b din {0, 1...., 8}, a mod 3 == 0 si b mod 3 == 0
#               unde Sab = {Xij | i,j numere intregi, i din [a, a+2], j din [b, b+2]}
#             C4: pentru i,j dati, Xij din {2, 4, 6, 8}
import copy
import time
from collections import deque


# Functia de initializare
def initialize(instance):
    size = len(instance)
    structure = [[{'value': instance[i][j],
                   'domain': instance[i][j] if instance[i][j] > 0 else list(range(1, 10)) if instance[i][j] == -1
                   else [2, 4, 6, 8]} for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            if structure[i][j]['value'] > 0:
                value = structure[i][j]['value']
                for y in range(size):
                    if y != j and structure[i][y]['value'] < 1 and value in structure[i][y]['domain']:
                        structure[i][y]['domain'].remove(value)
                for x in range(size):
                    if x != i and structure[x][j]['value'] < 1 and value in structure[x][j]['domain']:
                        structure[x][j]['domain'].remove(value)

    for i in range(0, size, 3):
        for j in range(0, size, 3):
            square_values = set()
            for x in range(i, i + 3):
                for y in range(j, j + 3):
                    if structure[x][y]['value'] > 0:
                        square_values.add(structure[x][y]['value'])
            for x in range(i, i + 3):
                for y in range(j, j + 3):
                    if structure[x][y]['value'] == -1:
                        structure[x][y]['domain'] = list(set(structure[x][y]['domain']) - square_values)
    for i in range(size):
        for j in range(size):
            print(i, j, structure[i][j]['domain'])
        print("\n")
    return structure


instance = [
    [8, 4, -1, -1, 5, -1, 0, -1, -1],
    [3, -1, -1, 6, -1, 8, -1, 4, -1],
    [-1, -1, 0, 4, -1, 9, -1, -1, 0],
    [-1, 2, 3, -1, 0, -1, 9, 8, -1],
    [1, -1, -1, 0, -1, 0, -1, -1, 4],
    [-1, 9, 8, -1, 0, -1, 1, 6, -1],
    [0, -1, -1, 5, -1, 3, 0, -1, -1],
    [-1, 3, -1, 1, -1, 6, -1, -1, 7],
    [-1, -1, 0, -1, 2, -1, -1, 1, 3]
]


def is_final(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]['value'] < 1:
                return False
    return True


def show(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]['value'] >= 1:
                print(sudoku[i][j]['value'], end=" | ")
            elif sudoku[i][j]['value'] == -1:
                print("\033[31m" + str(sudoku[i][j]['value']) + "\033[0m", end=" | ")
            else:
                print("\033[34m" + str(sudoku[i][j]['value']) + "\033[0m", end=" | ")
        print("\n-----------------------------------------")
    print("\n")


def transform(sudoku, row, col, value):
    new_sudoku = copy.deepcopy(sudoku)
    new_sudoku[row][col]['value'] = value
    new_sudoku[row][col]['domain'] = value
    for i in range(9):
        if i != col and new_sudoku[row][i]['value'] < 1 and value in new_sudoku[row][i]['domain']:
            new_sudoku[row][i]['domain'].remove(value)
            if len(new_sudoku[row][i]['domain']) == 0:
                print("\033[33mDomain is empty for " + str(row) + " " + str(i) + "\033[0m")
                return None
        if i != row and new_sudoku[i][col]['value'] < 1 and value in new_sudoku[i][col]['domain']:
            new_sudoku[i][col]['domain'].remove(value)
            if len(new_sudoku[i][col]['domain']) == 0:
                print("\033[33mDomain is empty for " + str(i) + " " + str(col) + "\033[0m")
                return None
    s_row = row // 3
    s_col = col // 3
    for i in range(s_row * 3, s_row * 3 + 3):
        for j in range(s_col * 3, s_col * 3 + 3):
            if (i != row or j != col) and new_sudoku[i][j]['value'] < 1 and value in new_sudoku[i][j]['domain']:
                new_sudoku[i][j]['domain'].remove(value)
                if len(new_sudoku[i][j]['domain']) == 0:
                    print("\033[33mDomain is empty for " + str(i) + " " + str(j) + "\033[0m")
                    return None
    return new_sudoku


# FORWARD CHECKING


def forward_checking(sudoku):
    if is_final(sudoku):
        return sudoku
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]['value'] < 1:
                print(i, j, sudoku[i][j]['domain'])
                for value in list(sudoku[i][j]['domain']):
                    print("\033[32mTrying value " + str(value) + " for cell " + str(i) + " " + str(j) + "\033[0m")
                    updated_sudoku = transform(sudoku, i, j, value)
                    if updated_sudoku is not None:
                        show(updated_sudoku)
                        result = forward_checking(updated_sudoku)
                        if result:
                            return result
                return None

    return None


# MRV + FORWARD CHECKING

def mrv(sudoku):
    min_domain = 10
    min_row = -1
    min_col = -1
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]['value'] < 1 and len(sudoku[i][j]['domain']) < min_domain:
                min_domain = len(sudoku[i][j]['domain'])
                min_row = i
                min_col = j
    return min_row, min_col


def forward_checking_mrv(sudoku):
    if is_final(sudoku):
        return sudoku
    row, col = mrv(sudoku)
    if row == -1 and col == -1:
        return sudoku
    for value in list(sudoku[row][col]['domain']):
        print("\033[32mTrying value " + str(value) + " for cell " + str(row) + " " + str(col) + "\033[0m")
        updated_sudoku = transform(sudoku, row, col, value)
        if updated_sudoku is not None:
            show(updated_sudoku)
            result = forward_checking_mrv(updated_sudoku)
            if result:
                return result
    return None


def measure(sudoku):
    print(" -----------------  FORWARD CHECKING  ----------------- ")
    start_time = time.time()
    solution = forward_checking(sudoku)
    end_time = time.time()
    exec_time = end_time - start_time
    print(" -----------------  FORWARD CHECKING + MRV  ----------------- ")
    start_time_mrv = time.time()
    solution_mrv = forward_checking_mrv(sudoku)
    end_time_mrv = time.time()
    exec_time_mrv = end_time_mrv - start_time_mrv
    print(" -----------------  ARC CONSISTENCY  ----------------- ")
    start_time = time.time()
    solution_ac = forward_checking_ac(sudoku)
    end_time = time.time()
    exec_time_ac = end_time - start_time
    if solution is not None:
        print("Solution found with forward checking in " + str(exec_time))
        show(solution)
    if solution_mrv is not None:
        print("Solution found with forward checking + MRV in " + str(exec_time_mrv))
        show(solution_mrv)
    if solution_ac is not None:
        print("Solution found with arc consistency in " + str(exec_time_ac))
        show(solution_ac)


initial_sudoku = initialize(instance)


# ARC CONSISTENCY


def get_adjacent(row, col):
    adj = []
    for i in range(9):
        adj.append((row, i))
        adj.append((i, col))
    s_row = (row // 3) * 3
    s_col = (col // 3) * 3
    for i in range(s_row, s_row + 3):
        for j in range(s_col, s_col + 3):
            adj.append((i, j))
    adj = list(set(adj))
    adj.remove((row, col))
    return adj


def generate_arcs(sudoku):
    arcs = []
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]['value'] < 1:
                for adj in get_adjacent(i, j):
                    if sudoku[adj[0]][adj[1]]['value'] < 1:
                        arcs.append(((i, j), adj))
    return arcs


def verify_consistent_arc(sudoku, cell1, cell2):
    row1, col1 = cell1
    row2, col2 = cell2
    new_sudoku = copy.deepcopy(sudoku)
    for value in new_sudoku[row1][col1]['domain']:
        ok = False
        for val in new_sudoku[row2][col2]['domain']:
            if value != val:
                ok = True
        if not ok:
            new_sudoku[row1][col1]['domain'].remove(value)
            print("\033[31:33mInconsistent arc" + str(cell1) + "-" + str(cell2) + " because of value:  " + str(
                value) + "\033[0m")
            if len(new_sudoku[row1][col1]['domain']) == 0:
                print("\033[33m" + str(cell1) + " doesn't have anymore values " + "\033[0m")
                return None
    return new_sudoku


def arc_consistency(sudoku):
    queue = deque()
    arcs = generate_arcs(sudoku)
    queue.extend(arcs)
    copy_sudoku = copy.deepcopy(sudoku)
    while queue:
        cell1, cell2 = queue.popleft()
        new_sudoku = verify_consistent_arc(copy_sudoku, cell1, cell2)
        if new_sudoku is None:
            return None
        if new_sudoku != copy_sudoku:
            copy_sudoku = new_sudoku
            queue.clear()
            queue.extend(arcs)
    return copy_sudoku


def forward_checking_ac(sudok):
    if is_final(sudok):
        return sudok
    sudoku = arc_consistency(sudok)
    for i in range(9):
        for j in range(9):
            if sudoku[i][j]['value'] < 1:
                for value in list(sudoku[i][j]['domain']):
                    print("\033[32mTrying value " + str(value) + " for cell " + str(i) + " " + str(j) + "\033[0m")
                    new_sudoku = transform(sudoku, i, j, value)
                    if new_sudoku is not None:
                        arc_cons = arc_consistency(new_sudoku)
                        show(new_sudoku)
                        if arc_cons is None:
                            continue
                        result = forward_checking_ac(arc_cons)
                        if result:
                            return result
                    return None

    return None


# solution = forward_checking(initial_sudoku)
# show(solution)
# solution2 = forward_checking_mrv(initial_sudoku)
# show(solution2)
# solution3 = forward_checking_ac(initial_sudoku)
# show(solution3)
#measure(initial_sudoku)

instance2 = [
    [-1, 0, -1, 4, -1, -1, 8, -1, 0],
    [5, 2, 0, 0, -1, 8, -1, -1, -1],
    [9, -1, 0, 5, 0, -1, 0, 3, 6],
    [0, 0, -1, -1, -1, 0, 0, -1, -1],
    [6, -1, -1, -1, 0, 0, -1, 9, 0],
    [0, -1, 1, -1, -1, 0, 2, 8, -1],
    [8, 0, 0, 9, 7, -1, -1, 6, 3],
    [-1, -1, 3, 0, 6, -1, -1, 0, 0],
    [-1, 5, 0, 0, 0, 3, -1, 0, 1]
]

initial_sudoku2 = initialize(instance2)
measure(initial_sudoku2)