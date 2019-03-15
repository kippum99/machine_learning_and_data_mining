import os
import random

text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()

# Build a probability table for punctuations

lines = [line.split() for line in text.split('\n') if line.split()]
sonnets = []
sonnet = []
for line in lines:
    if len(line) == 1:
        # Only store sonnets with 14 lines
        if len(sonnet) == 14:
            sonnets.append(sonnet)
        sonnet = []
        continue
    sonnet.append(line)

# P (punc | location in poem)
# punc_probs[i][j] returns P (punc = j | loc = i)
punc_probs = [[0 for c in range(7)] for r in range(4)]

puncs = [',', '.', ':', ';', '?', '!']
loc1 = [0, 1, 2, 4, 5, 6, 8, 9, 10] # 1st-3rd lines of 1st-3rd stanzas
loc2 = [3, 7, 11] # 4th line of 1st-3rd stanzas

for sonnet in sonnets:
    for i in range(14):
        punc = sonnet[i][-1].strip("'").strip(')')[-1]
        if punc in puncs:
            punc_index = puncs.index(punc)
        # No punctuation
        else:
            punc_index = 6
        if i in loc1:
            punc_probs[0][punc_index] += 1
        elif i in loc2:
            punc_probs[1][punc_index] += 1
        elif i == 12:
            punc_probs[2][punc_index] += 1
        elif i == 13:
            punc_probs[3][punc_index] += 1

# Convert freq to prob
for loc in range(4):
    sum_probs = sum(punc_probs[loc])
    punc_probs[loc] = [prob / sum_probs for prob in punc_probs[loc]]


# Sample punctuation given a location in poem
def get_punc(loc):
    probs = punc_probs[loc]
    rand_var = random.uniform(0, 1)
    next_punc = 0
    while rand_var > 0:
        rand_var -= probs[next_punc]
        next_punc += 1
    punc = next_punc - 1
    if punc == 6:
        return ''
    else:
        return puncs[punc]
