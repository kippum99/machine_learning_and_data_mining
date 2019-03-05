import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err


def main():
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

    k = 20
    #regularization constants
    regs = [10**-4, 10**-3, 10**-2, 10**-1, 1]
    #learning rate
    eta = 0.01
    #0.00005 best
    epsilons = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002]
    E_ins = []
    E_outs = []

    # Use to compute Ein and Eout
    for reg in regs:
        E_ins_for_lambda = []
        E_outs_for_lambda = []

        for ep in epsilons:
            print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s, ep = %s"%(M, N, k, eta, reg, ep))
            U, V, e_in = train_model(M, N, k, eta, reg, Y_train, ep)
            E_ins_for_lambda.append(e_in)
            eout = get_err(U, V, Y_test)
            E_outs_for_lambda.append(eout)

        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)

    for i in range(len(regs)):
        plt.plot(epsilons, E_ins[i], label='$E_{in}, \lambda=$'+str(regs[i]))
    plt.title('$E_{in}$ vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.legend()
    plt.savefig('E_in.png')
    plt.clf()

    for i in range(len(regs)):
    	plt.plot(epsilons, E_outs[i], label='$E_{out}, \lambda=$'+str(regs[i]))
    plt.title('$E_{out}$ vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.legend()
    plt.savefig('E_out.png')

if __name__ == "__main__":
    main()
