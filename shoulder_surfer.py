import pandas as pd
import numpy as np
from tqdm import tqdm
import dataframe_image as dfi


def gen_combinations(k, n):
    """
    Generate all possible compositions.
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
    """
    Calculate level for probability.
    """
    return np.floor(0.5 - alpha * np.log(p)) - beta


def sneaky_surfer(v, transition_matrix, l, pin_length, n=3):
    """
    Calculate most likely PINs inspired by the algorithm of Dürmuth et al. (2015).
    :param v: first observed digit.
    :return: the n most likely PINs.
    """

    highest_prob = np.max(transition_matrix)
    lowest_prob = np.min(transition_matrix)

    # alpha, beta are necessary to calculate the level of the probability
    alpha, beta = calc_alpha_beta(highest_prob, lowest_prob, l)

    # apply the level function to each element in the transition matrix
    level_matrix = level(transition_matrix, alpha, beta)

    rank = pin_length    # minimum possible rank is pin_length
    pins = []
    while len(pins) < n:  # stop if we have the desired number of PINs

        # get all compositions of rank with composition is equal to PIN length - 1
        combis = list(gen_combinations(pin_length-1, rank))
        for combi in combis:
            results = []
            traverse_tree(combi, [v], level_matrix, results)
            pins.extend(results)
        rank += 1
    return pins


def traverse_tree(combi, path, level_matrix, results):
    """
    Iterate through the tree (markov model graph) from the start digit v until a path is found which matches the levels
    of the composition combi.
    :param combi: composition
    :param path: possible PIN - growing through recursion
    :param level_matrix: transformed transition matrix with level function
    :param results: likely PINs
    :return:
    """
    if len(path) == (len(combi)+1):
        results.append(path)
        return

    for j, val in enumerate(level_matrix[path[-1]]):
        if val == combi[len(path)-1]:
            traverse_tree(combi, np.append(path, j), level_matrix, results)


def calc_alpha_beta(ph, pl, l):
    alpha = (1-l) / (np.floor(0.5-np.log(ph)) - np.floor(0.5-np.log(pl)))
    beta = np.floor(0.5 - alpha * np.log(ph)) - 1
    return alpha, beta


def print_matrix(m):
    """
    Export matrix as png with colored entries by its gradient.
    :param m: Matrix
    :return: void
    """
    digit_names = [f'Digit {i}' for i in range(10)]
    df = pd.DataFrame(m, columns=digit_names, index=digit_names)
    df_styled = df.style.background_gradient()
    dfi.export(df_styled, 'df_styled.png')
    df.dfi.export('df.png')


def generate_transition_matrix(four_digit=True):
    """
    Load either 4-PINs or 6-PINs dataset and create first order markov models on that basis.
    :param four_digit: boolean
    :return: transition matrix
    """
    filename = "RockYou-4-digit.txt" if four_digit else "RockYou-6-digit.txt"
    file = open(filename, 'r')
    lines = file.readlines()

    transition_matrix = np.zeros((10, 10))
    for line in tqdm(lines):
        for j, digit in enumerate(line[:-2]):   # slice '\n' and last character
            # digit represents the current state and int(line[j+1]) the next state
            transition_matrix[int(digit)][int(line[j+1])] += 1

    transition_matrix = transition_matrix / np.expand_dims(np.sum(transition_matrix, axis=1), axis=1)
    assert np.all(np.isclose(np.ones(10), np.sum(transition_matrix, axis=1))), "Sum of rows should be 1.0!"

    return transition_matrix


def print_pins(pins):
    for pin in pins:
        print(*pin, sep=' ')


if __name__ == '__main__':
    l = 100  # highest level representing the smallest transition probability
    observed_pin = 2
    n_likely_pins = 3

    # Start with the 4-PIN dataset.
    pin_length = 4
    transition_matrix = generate_transition_matrix(four_digit=True)
    pins = sneaky_surfer(observed_pin, transition_matrix, l, pin_length, n_likely_pins)
    print('The most likely PINs are:')
    print_pins(pins)

    # Continue with the 6-PIN dataset.
    pin_length = 6
    transition_matrix = generate_transition_matrix(four_digit=False)
    pins = sneaky_surfer(observed_pin, transition_matrix, l, pin_length, n_likely_pins)
    print('The most likely PINs are:')
    print_pins(pins)

