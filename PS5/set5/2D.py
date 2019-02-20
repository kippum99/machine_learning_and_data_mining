# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
		
def main():
    Y_train = np.loadtxt('../data/train.txt').astype(int)
    Y_test = np.loadtxt('../data/test.txt')	.astype(int)
	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    Ks = [10,20,30,50,100]
	
    reg = 0.0
    eta = 0.03 # learning rate
    E_in = []
    E_out = []
	
    # Use to compute Ein and Eout
    for K in Ks:
        U,V, err = train_model(M, N, K, eta, reg, Y_train)
        E_in.append(err)
        E_out.append(get_err(U, V, Y_test))
	
    plt.plot(Ks, E_in, label='$E_{in}$')
    plt.plot(Ks, E_out, label='$E_{out}$')
    plt.title('Error vs. K')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('2d.png')	

   

if __name__ == "__main__":
    main()
