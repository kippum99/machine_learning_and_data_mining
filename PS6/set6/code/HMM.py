########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # Compute length-1 probs and seqs
        for a in range(self.L):
            probs[1][a] = self.O[a][x[0]] * self.A_start[a]
            seqs[1][a] = str(a)

        # Recursively solve for each j > 1
        for j in range(2, M + 1):
            for a in range(self.L):
                # Enumerate over previous probs and seqs
                # a is current state, b is previous state
                # Find prefix of length j - 1 maximizing probability of prefix
                # of length j ending in state a and store the prob and prefix
                prefix = ''
                max_prob = 0
                for b in range(self.L):
                    prob = probs[j-1][b] * self.A[b][a] * self.O[a][x[j-1]]
                    if prob >= max_prob:
                        prefix = seqs[j-1][b]
                        max_prob = prob
                probs[j][a] = max_prob
                seqs[j][a] = prefix + str(a)

        # Return max-probability state sequence of length M
        return seqs[M][probs[M].index(max(probs[M]))]


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Compute alphas for i = 1
        for a in range(self.L):
            alphas[1][a] = self.O[a][x[0]] * self.A_start[a]

        # Recursively solve for each i > 1
        for i in range(2, M + 1):
            for a in range(self.L):
                sum_probs = 0
                for b in range(self.L):
                    sum_probs += alphas[i-1][b] * self.A[b][a]
                alphas[i][a] = self.O[a][x[i-1]] * sum_probs
            if normalize:
                alphas[i] = [alpha / sum(alphas[i]) for alpha in alphas[i]]

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Compute betas for i = M
        betas[M] = [1. for _ in range(self.L)]

        # Recursively solve for each i < M
        for i in range(M - 1, 0, -1):
            for a in range(self.L):
                sum_probs = 0
                for b in range(self.L):
                    sum_probs += betas[i+1][b] * self.A[a][b] * self.O[b][x[i]]
                betas[i][a] = sum_probs
            if normalize:
                betas[i] = [beta / sum(betas[i]) for beta in betas[i]]

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        N = len(X)

        # Calculate each element of A using the M-step formulas.
        for a in range(self.L):
            for b in range(self.L):
                denom = 0
                numer = 0
                for i in range(N):
                    # Start from 1, since y^0, start state is always START
                    # and we're not interested in transition from y^0 to y^1
                    for j in range(1, len(Y[i])):
                        if Y[i][j-1] == a:
                            denom += 1
                            if Y[i][j] == b:
                                numer += 1
                self.A[a][b] = numer / denom

        # Calculate each element of O using the M-step formulas.
        for a in range(self.L):
            for w in range(self.D):
                denom = 0
                numer = 0
                for i in range(N):
                    for j in range(len(Y[i])):
                        if Y[i][j] == a:
                            denom += 1
                            if X[i][j] == w:
                                numer += 1
                self.O[a][w] = numer / denom


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # Number of training examples (sequences) in X
        N = len(X)

        # Store marginal probabilities in a matrix
        # marg2[i][j][a] = P(y_i^j = a, x_i) same shape as X
        # marg3[i][j][a][b] = P(y_i^j = a, y_i^(j+1) = b, x_i)
        # 2nd dimesnion of marg3 is M_i - 1, not M_i
        marg2 = [[[0. for _ in range(self.L)] for _ in range(len(seq))]
                                                                for seq in X]
        marg3 = [[[[0. for _ in range(self.L)] for _ in range(self.L)]
                    for _ in range(len(seq) - 1)] for seq in X]

        # Train for N_iters iterations
        for iter in range(N_iters):
            print(iter)
            # E-step - calculate marginal probabilities for each training example
            for i in range(N):
                alphas = self.forward(X[i], True)
                betas = self.backward(X[i], True)
                for j in range(1, len(X[i]) + 1):
                    alphas_j = alphas[j]
                    betas_j = betas[j]

                    # Compute marg2_ij_prob for each state a
                    # marg2_ij is a length-L vector of those probabilities with each
                    # element corresponding to each state
                    # Divide the vector by the sum of the elements (alpha * beta)
                    # (denominator is that sum for all probabilities)
                    marg2_ij = [0. for _ in range(self.L)]
                    for a in range(self.L):
                        marg2_ij[a] = alphas_j[a] * betas_j[a]
                    sum_denom = sum(marg2_ij)
                    marg2[i][j-1] = [prob / sum_denom for prob in marg2_ij]

                    # If j is the last index, j + 1 is out of bounds and marg3_ij
                    # cannot be computed
                    if j == len(X[i]):
                        break

                    # Compute marg3_ij_prob for each state a and b
                    # marg3_ij is a L x L matrix of those probabilities with each
                    # element corresponding to a pair of states a, b
                    marg3_ij = [[0. for _ in range(self.L)] for _ in range(self.L)]
                    sum_denom = 0
                    for a in range(self.L):
                        for b in range(self.L):
                            marg3_ij[a][b] = (alphas_j[a] * self.O[b][X[i][j]]
                                                * self.A[a][b] * betas[j+1][b])
                            sum_denom += marg3_ij[a][b]
                    marg3[i][j-1] = [[prob / sum_denom for prob in row]
                                                            for row in marg3_ij]

            # M-step - update A
            for a in range(self.L):
                for b in range(self.L):
                    denom = 0
                    numer = 0
                    for i in range(N):
                        for j in range(1, len(X[i])):
                            denom += marg2[i][j-1][a]
                            numer += marg3[i][j-1][a][b]
                    self.A[a][b] = numer / denom

            # M-step - update O
            for a in range(self.L):
                for w in range(self.D):
                    denom = 0
                    numer = 0
                    for i in range(N):
                        for j in range(len(X[i])):
                            denom += marg2[i][j][a]
                            if X[i][j] == w:
                                numer += marg2[i][j][a]
                    self.O[a][w] = numer / denom


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        # Initialize y^0 by randomly choosing a state (assume uniform dist.)
        states.append(random.randrange(self.L))
        emission.append(random.choices(range(self.D),
                                        weights=self.O[states[0]])[0])

        # Recursively sample
        for i in range(1, M):
            # Sample y^i from P(y^i | y^(i-1))
            states.append(random.choices(range(self.L),
                                            weights=self.A[states[i-1]])[0])

            # Sample x^i from P(x^i | y^i)
            emission.append(random.choices(range(self.D),
                                            weights=self.O[states[i]])[0])

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(2019)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
