
import numpy as np
import operator
import sys
import cvxpy as cp
from collections import defaultdict
import streamlit as st
import os
from cvxpy.reductions.solvers.defines import installed_solvers
from cvxpy import GLPK
import cvxopt
# from gurobipy import *


def expand(pop_selector, pop_dict):
    pops = pop_selector.split('+')
    ret = []
    for pop in pops:
        ret += pop_dict.get(pop, pop)
    return ret


def distance(M, b):
    b = b[:, np.newaxis]
    return np.sqrt(np.sum((M - b) ** 2, axis=0))


# def find_nearest_to_avg(M, avgpop, indiv2index, poplist):
#     min_distance = float('inf')
#     closest = None
#     for p in poplist:
#         distance = np.linalg.norm(M[:, indiv2index[p]]-avgpop)
#         if distance < min_distance:
#             min_distance = distance
#             closest = p
#     return M[:, indiv2index[closest]]


# def distances_to_convex_combinations(M, b, indiv2index, pop_dict, threshold=.00001):
#     pop2fit = []
#     for pop, indiv_list in pop_dict.items():
#         Msub = M[:, [indiv2index[indiv] for indiv in indiv_list]]
#         x = cp.Variable(Msub.shape[1])
#         cost = cp.norm2(Msub @ x - b)**2
#         constraints = [cp.sum(x) == 1, 0 <= x]
#         l = []
#         print(pop)

#         prob = cp.Problem(cp.Minimize(cost), constraints)
#         prob.solve()
#         dindiv = defaultdict(int)

#         for i, _ in enumerate(range(Msub.shape[1])):
#             dindiv[indiv_list[i]] += x.value[i]
#         residual_norm = cp.norm(Msub @ x - b, p=2).value
#         vector = Msub @ x.value
#         print(Msub.shape)
#         print(x.shape)

#         print('-------------- ANCESTRY BREAKDOWN: -------------')
#         for k, v in dindiv.items():
#             l.append((k, v))
#         l_sort = sorted(l, key=lambda x: -x[1])
#         for x in l_sort:
#             if x[1] < threshold:
#                 break
#             print(f'{x[0]: <50}--->\t{x[1]*100:.3f}%')
#         print('------------------------------------------------')
#         print(f'Fit error: {residual_norm}')
#         pop2fit.append((pop, residual_norm, vector))
#         print()
#         print()
#     print(pop2fit)
#     pop2fit_sort = sorted(pop2fit, key=lambda x: x[1])
#     with open('out.txt', 'w') as f:
#         f.write(',PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16,PC17,PC18,PC19,PC20,PC21,PC22,PC23,PC24,PC25\n')
#         for pop, error, vector in pop2fit_sort:
#             print(f'{pop}---->{error}---->{vector}')
#             f.write(f'{pop},%s\n' % ','.join(map(str, list(vector))))

# Based on https://github.com/michal3141/g25
def main(coords, file_names, penalty, nonzeros):
    m = []
    l = []
    index2pop = []
    index2indiv = []
    indiv2index = {}
    pop2percent = []
    # penalty = 0.
    # Setting the penalty to 0.01.
    # penalty = 0.01
    noise_penalty = 0.
    indiv = ''
    threshold = .00001
    constraint_dict = {}
    operator_dict = {}
    pop_dict = defaultdict(list)
    # nonzeros = 0

    # sheetfile = sys.argv[1]
    # indivfile = sys.argv[2]
    # Creating a file path to the selected file.
    sheetfile = f'pages/models/{selected_file}.txt'

    for arg in sys.argv[3:]:
        for operator in ['<=', '>=', '=']:
            if operator in arg:
                arg_splitted = arg.split(operator)
                pop_selector = arg_splitted[0]
                pen = float(arg_splitted[1])
                op = operator
                break
        if pop_selector.startswith('pen'):
            if pop_selector == 'pen':
                penalty = pen
            else:
                raise NotImplementedError(
                    "Penalizing individuals or populations is not implemented yet.")
        elif pop_selector.startswith('count'):
            nonzeros = int(pen)
        else:
            constraint_dict[pop_selector] = pen
            operator_dict[pop_selector] = op

    with open(sheetfile, 'r') as f:
        f.readline()
        for index, line in enumerate(f):
            arr = line.strip().split(',')
            indivname = arr[0]
            ethname = arr[0].split(':')[0]
            index2pop.append(ethname)
            index2indiv.append(indivname)
            indiv2index[indivname] = index
            pop_dict[ethname].append(indivname)
            m.append(np.array([float(x) for x in arr[1:]]))

    M = np.column_stack(m)

    data = ['PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16,PC17,PC18,PC19,PC20,PC21,PC22,PC23,PC24,PC25']
    data.append(coords)

    for line in data[1:]:
        arr = line.strip().split(',')
        indiv = arr[0]
        b = np.array([float(x) for x in arr[1:]])

    x = cp.Variable(M.shape[1])
    cost = cp.norm2(M @ x - b)**2 + penalty * \
        cp.sum(cp.multiply(distance(M, b), x))

    constraints = [cp.sum(x) == 1, 0 <= x]

    for pop_selector, pen in constraint_dict.items():
        op = operator_dict[pop_selector]
        sum_expr = cp.sum([x[indiv2index[p]]
                          for p in expand(pop_selector, pop_dict)])
        if op == '=':
            constraints.append(sum_expr == pen)
        elif op == '>=':
            constraints.append(sum_expr >= pen)
        elif op == '<=':
            constraints.append(sum_expr <= pen)

    if int(nonzeros) > 0:
        binary = cp.Variable(M.shape[1], boolean=True)
        constraints += [x - binary <= 0., cp.sum(binary) == nonzeros]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    # prob.solve(solver=GLPK, verbose=True)
    prob.solve(verbose=True)
    # prob.solve(verbose=True)
    dindiv = defaultdict(int)
    dpop = defaultdict(int)

    for i, _ in enumerate(range(M.shape[1])):
        dindiv[index2indiv[i]] += x.value[i]
        dpop[index2pop[i]] += x.value[i]
    residual_norm = cp.norm(M @ x - b, p=2).value
    print('-------------- ANCESTRY BREAKDOWN: -------------')
    for k, v in dindiv.items():
        l.append((k, v))
    l_sort = sorted(l, key=lambda x: -x[1])
    for x in l_sort:
        if x[1] < threshold:
            break
        print(f'{x[0]: <50}--->\t{x[1]*100:.3f}%')
        # with st.spinner('Loading...'):
        c.write(f'{x[0]: <50}--->\t{x[1]*100:.1f}%')
        # c.write(f'{x[0]: <50}--->\t{round(x[1]*100,1)}%')
    print('------------------------------------------------')
    print(f'Fit error: {residual_norm}')


solvers = installed_solvers()
print(f"Installed solvers: {solvers}")

# Creating a list of files in the models directory and then creating a dropdown menu to select one of
# the files.
dir_path = "pages/models"
file_names = [f for f in os.listdir(
    dir_path) if os.path.isfile(os.path.join(dir_path, f))]
base_names = [os.path.splitext(name)[0] for name in file_names]
selected_file = st.selectbox("Select a model", base_names)

# Display the selected file name
st.write(f"Selected model: {selected_file}")
coords = st.text_input('G25 Coordinates', '')
# create streamlit checkbox to enable penalty
col1, col2 = st.columns(2)
pen = False
nonzeros = 0
print("Penalty is disabled")
with col1:
    penalty = st.checkbox('Enable penalty')
with col2:
    reduce_pop = st.checkbox('Reduce population')

if reduce_pop:
    nonzeros = st.selectbox(
        'Number of populations',
        ('3', '4', '5'))
    # print(f"Number of populations: {option}")

if penalty:
    penalty = 0.01
    print("Penalty is enabled")
    st.write("Penalty is enabled")

c = st.container()
c.button('Calculate', on_click=main, args=(
    coords, selected_file, penalty, nonzeros))


# if __name__ == '__main__':
#     main()
