import os

def get_syllable_dict():
    '''
    Takes in no input and returns a dictionary with keys as all the words in
    Shakespeare's sonnets and values as [real, end], where real is a list of
    possible real syllable counts for the corresponding word and end is a list
    of possible end syllable counts (when the word is at the end of a line).
    '''
    file = open(os.path.join(os.getcwd(),
                'data/Syllable_dictionary.txt')).read()
    lines = [line.split() for line in file.split('\n') if line.split()]

    syllable_dict = {}

    for line in lines:
        real = []
        end = []
        for i in range(1, len(line)):
            if line[i][0] == 'E':
                end.append(int(line[i][1]))
            else:
                real.append(int(line[i][0]))
        syllable_dict[line[0]] = [real[::-1], end[::-1]]

    return syllable_dict
