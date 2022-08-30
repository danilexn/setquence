import numpy as np


def migrate(units, j, schedule_messages):
    if units == 0:
        return
    schedule_messages[j] += units


def distribute(units, N_i, schedule_messages):
    if units == 0:
        return
    for j in N_i:
        schedule_messages[j] += units / len(N_i)


def dasud(rank, neighbors, random_load, schedule_messages):
    loads = np.array(random_load)
    i = rank
    N_i = np.array(neighbors)
    index_neighbors = np.where(N_i != i)
    N_i = N_i[index_neighbors]
    index_neighbors = N_i
    L = np.array(loads)
    L_bar = np.sum(L) / (len(L))
    E_i = L[i] - L_bar
    N_i_low = N_i[np.where(L[index_neighbors] < L_bar)]

    L_difi = np.max(L[index_neighbors]) - np.min(L[index_neighbors])
    L_vi = np.sum(L[index_neighbors]) / len(index_neighbors)
    rho_v = np.sum(np.sqrt((L_vi - L[index_neighbors]) ** 2))
    L_sum_neigh = np.sum(L_bar - L[N_i_low])

    if len(N_i_low) != 0 and E_i > 0:
        for j in N_i_low:
            P_i_j = (L_bar - L[j]) / L_sum_neigh
            l_send = int(E_i * P_i_j)
            migrate(l_send, j, schedule_messages)
    elif L_difi > 1:
        if rho_v == 0:
            distribute(L[i] - L_vi + 1, N_i, schedule_messages)
        else:
            migrate(1, N_i[np.argmin(L[index_neighbors])], schedule_messages)
