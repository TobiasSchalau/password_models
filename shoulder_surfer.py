import pandas as pd
import numpy as np
from tqdm import tqdm
import dataframe_image as dfi
import six


def gen_combinations(k, n):
    """
    Imported from https://stackoverflow.com/questions/49200566/function-for-compositions-in-python
    """
    assert n > k > 1
    to_process = [[i] for i in range(1, n+1)]
    while to_process:
        l = to_process.pop()
        s = sum(l)
        le = len(l)
        #If you do not distiguish permutations put xrange(l[-1],n-s+1)
        # instead, to avoid generating repeated cases.
        # And if you do not want number repetitions, putting
        # xrange(l[-1] + 1, n-s+1) will do the trick.
        for i in range(1, n-s+1):
            news = s + i
            if news <= n:
                newl = list(l)
                newl.append(i)
                if le == k-1 and news == n:
                    yield tuple(newl)
                elif le < k-1 and news < n:
                    to_process.append(newl)


def level(p, alpha, beta):
    return np.floor(0.5 - alpha * np.log(p)) - beta


def sneaky_surfer(v, transition_matrix, l, n=3):
    """
    :param v: first digit of the pin
    :return: the n most likely PINs
    """

    highest_prob = np.max(transition_matrix)
    lowest_prob = np.min(transition_matrix)

    alpha, beta = calc_alpha_beta(highest_prob, lowest_prob, l)
    level_matrix = level(transition_matrix, alpha, beta)

    rank = 4
    pins = []
    while len(pins) < n:
        combis = list(gen_combinations(3, rank))
        for combi in combis:
            result = traverse_tree(combi, [v], level_matrix)
            if result is not None:
                pins.append(result)
        rank += 1
    return pins


def traverse_tree(combi, path, level_matrix):
    if len(path) == (len(combi)+1):
        return path

    for j, val in enumerate(level_matrix[path[-1]]):
        if val == combi[len(path)-1]:
            return traverse_tree(combi, np.append(path, j), level_matrix)


def calc_alpha_beta(ph, pl, l):
    alpha = (1-l) / (np.floor(0.5-np.log(ph)) - np.floor(0.5-np.log(pl)))
    beta = np.floor(0.5 - alpha * np.log(ph)) - 1
    return alpha, beta


if __name__ == '__main__':
    """
    Start with the 4-PIN dataset.
    """
    file = open("RockYou-4-digit.txt", 'r')
    lines = file.readlines()
    transition_matrix = np.zeros((10, 10))
    for line in tqdm(lines):
        for j, digit in enumerate(line[:-2]):   # slice '\n' and last character
            # digit represents the current state and int(line[j+1]) the next state
            transition_matrix[int(digit)][int(line[j+1])] += 1
    #print(transition_matrix)

    transition_matrix = transition_matrix / np.expand_dims(np.sum(transition_matrix, axis=1), axis=1)

    #print(transition_matrix)

    assert np.all(np.isclose(np.ones(10), np.sum(transition_matrix, axis=1))), "Sum of rows should be 1.0!"

    digit_names = [f'Digit {i}' for i in range(10)]

    """
    df = pd.DataFrame(transition_matrix, columns=digit_names, index=digit_names)
    df_styled = df.style.background_gradient()
    print(df_styled)
    dfi.export(df_styled, 'df_styled.png')
    df.dfi.export('df.png')
    """
    l = 100

    pins = sneaky_surfer(2, transition_matrix, l)

    print(pins)
