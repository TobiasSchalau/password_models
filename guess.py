import numpy as np
import itertools
import pandas as pd
import dataframe_image as dfi

pins = [1331,2303,1301,2320,1312,1330,1203,1033,2332,2323]
pins = list(map(lambda x: str(x) + 'T', pins))

transition_matrix = np.zeros((5, 5))
for pin in pins:
    for j, digit in enumerate(pin[:-1]):

        # digit represents the current state and int(line[j+1]) the next state
        if pin[j+1] == 'T':
            transition_matrix[int(digit)][-1] += 1
            continue

        transition_matrix[int(digit)][int(pin[j+1])] += 1

transition_matrix = transition_matrix / \
    np.expand_dims(np.sum(transition_matrix, axis=1), axis=1)

combinations = [p for p in itertools.product([0,1,2,3], repeat=3)]

probs = []
for comb in combinations:
    p = transition_matrix[1][comb[0]]
    for i in range(len(comb)-1):
        p*=transition_matrix[comb[i]][comb[i+1]]
    p*=transition_matrix[comb[-1]][-1]
    probs.append(p)

ranked = np.array(list(reversed(np.argsort(probs))))

df = pd.DataFrame([np.array(probs)[ranked], \
    np.array(combinations)[ranked]]).T
df.columns=['probability', 'combi']
print(df)
df.dfi.export('prob.png')