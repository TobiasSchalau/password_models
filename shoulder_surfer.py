import pandas as pd
import numpy as np
from tqdm import tqdm
import dataframe_image as dfi

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
    df_styled = df.style.background_gradient()
    print(df_styled)
    dfi.export(df_styled, 'df_styled.png')
    df.dfi.export('df.png')
    #df_styled.export_png('df_styled.png')
