import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.plotting import table
import dataframe_image as dfi
import six


def level(p, alpha, beta):
    return np.floor(0.5 - alpha * np.log(p)) - beta


def sneaky_surfer(v, n=3):
    """
    :param v: first digit of the pin
    :return: the n most likely PINs
    """
    rank = 1


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
    print(transition_matrix)

    transition_matrix = transition_matrix / np.expand_dims(np.sum(transition_matrix, axis=1), axis=1)

    print(transition_matrix)

    assert np.all(np.isclose(np.ones(10), np.sum(transition_matrix, axis=1))), "Sum of rows should be 1.0!"

    digit_names = [f'Digit {i}' for i in range(10)]
    df = pd.DataFrame(transition_matrix, columns=digit_names, index=digit_names)

    highest_prob = np.max(transition_matrix)
    lowest_prob = np.min(transition_matrix)
    l = 100

    alpha, beta = calc_alpha_beta(highest_prob, lowest_prob, l)
    print(alpha, beta)
    #print(level(highest_prob, alpha,beta), level(lowest_prob, alpha, beta))
    #print(level(0.25, alpha, beta))
    #print(level(0.2, alpha, beta))
    #print(level(0.05, alpha, beta))

