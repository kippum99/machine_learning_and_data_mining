import numpy as np

def grad_U(Ui, Yij, Vj, ai, bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), bias term ai for ith user, bias term bj for
    jth movie, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - Vj * (Yij - np.dot(Ui, Vj) - ai - bj))

def grad_V(Vj, Yij, Ui, ai, bj, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), bias term ai for ith user, bias term bj for
    jth movie, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - Ui * (Yij - np.dot(Ui, Vj) - ai - bj))

def grad_bias(Ui, Vj, Yij, ai, bj, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), bias term ai for ith user, bias term bj for
    jth movie, and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to ai (or bj, which is the same with a single data point)
    multiplied by eta.
    """

    return eta * -1 * (Yij - np.dot(Ui, Vj) - ai - bj)

def get_err(U, V, a, b, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column
    of V^T added to bias terms ai and bj.
    """
    tot_err = 0

    for row in Y:
        [i, j, y] = row
        tot_err += 0.5 * (y - np.dot(U[i-1], V[j-1]) - a[i-1] - b[j-1]) ** 2

    return tot_err / len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V, and bias terms a's and b's such that
    rating Y_ij is approximated by (UV^T)_ij + ai + bj.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, a, b, err) consisting of U, V, a, b, and the
    unregularized MSE of the model.
    """
    # Initialize U, V, a, b
    U = np.random.uniform(-0.5, 0.5, (M, K))
    V = np.random.uniform(-0.5, 0.5, (N, K))
    a = np.random.uniform(-0.5, 0.5, (M, ))
    b = np.random.uniform(-0.5, 0.5, (N, ))

    # Store initial loss reduction
    init_loss_reduction = get_err(U, V, a, b, Y)

    for epoch in range(max_epochs):
        # Shuffle training data indices
        indices = np.random.permutation(len(Y))

        # Loop through each data point and update weights
        for index in indices:
            i, j = Y[index][0] - 1, Y[index][1] - 1
            y = Y[index][2]
            U[i] -= grad_U(U[i], y, V[j], a[i], b[j], reg, eta)
            V[j] -= grad_V(V[j], y, U[i], a[i], b[j], reg, eta)
            grad_a_b = grad_bias(U[i], V[j], y, a[i], b[j], eta)
            a[i] -= grad_a_b
            b[j] -= grad_a_b

        # Store initial loss reduction or compare with intiial loss reduction
        loss = get_err(U, V, a, b, Y)
        if epoch == 0:
            init_loss_reduction -= loss
            prev_loss = loss
        else:
            if (prev_loss - loss) / init_loss_reduction <= eps:
                break
            prev_loss = loss

    return U, V, a, b, get_err(U, V, a, b, Y)
