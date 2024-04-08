import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.optimize import minimize

class ISS_GPR:
    # Constructor
    def __init__(self, X_train, y_train, hyper_param, D, dim_x = 1, dim_y = 1):

        # Initial hyper parameter: l, sigma_f, sigma_n
        self.hyper_param = hyper_param

        # Puts hyper parameter
        # l: characteristic length
        # sigma_f: signal noise
        # signa_n: sensor noise
        self.l = hyper_param[0]
        self.sigma_f = hyper_param[-2]
        self.sigma_n = hyper_param[-1]

        # Training data:
        # X_train: n samples with p dim
        # Y_train: n samples with q dim
        self.X_train = X_train
        self.y_train = y_train

        # Dimension info
        self.dim_x = dim_x
        self.dim_y = dim_y

        # The number of features
        self.D = D

        # Sets noise matrix (dim: 2d x 2d)
        self.R = self.sigma_n*np.diag(np.ones(2*self.D))

        # Initializes weight
        self.weight = np.zeros((2*self.D,1))

        # Sets mean value and kernel
        self.mu = np.zeros((self.D,self.dim_x))
        self.Sigma = 1/2*np.diag(1/(self.l)**2*np.ones((self.D,1)))

        # Generate random variable (dim: D x p)
        self.omega = np.random.normal(self.mu, self.Sigma, (self.D, dim_x))

        # A_0 = R'*R = R*R
        self.A = self.R.dot(self.R)

        # Initializes b vector
        self.b = np.zeros((2*self.D,1))

        self.Phi = np.zeros((2*self.D,1))


    def Phi_map(self, x_new):
        # omega in D x n
        # x_new in n x 1
        # trig_input in D x 1
        # Feature Phi in 1 x 2D
        trig_input = self.omega.dot(x_new)
        cos_trig_input_array = np.cos(trig_input)
        sin_trig_input_array = np.sin(trig_input)
        self.Phi = self.sigma_f/np.sqrt(self.D)*np.hstack((cos_trig_input_array, sin_trig_input_array))
        self.Phi = self.Phi[:,np.newaxis]


    def update(self, x_new, y_new):
        """
        Update v vec, b vec, R, and weight
        :param x_new: input data
        :param y_new: output data
        """
        # Do feature mapping from new input data
        self.Phi_map(x_new)
        # Get new output data
        y_new = y_new[:,np.newaxis]

        # Rank 1 update : A_t = A_{t-1} + Phi*Phi^T
        self.A = self.A + self.Phi.dot(self.Phi.T)

        # Cholesky update(R,Phi) : R^T*R = A_t --> Get R
        self.R = np.linalg.cholesky(self.A).T

        # Vector b is updated
        self.b = self.b + self.Phi.dot(y_new)

         # weight = A_{t}^{-1}*b = (R^T*R)^{-1}*b
        # b = R^T*R*weight
        # Let R*weight = z, then b = R^T*z --> solve z
        z = np.linalg.solve(self.R.T, self.b)

        # z = R*weight --> solve weight
        self.weight = np.linalg.solve(self.R, z)

    def predict(self, x_new):
        # feature mapping
        self.Phi_map(x_new)
        # 1. predicted value = weight^T*Phi
        # 2. var = signma_n^2 * (1 + Phi^T*A^{-1}*Phi)
        # Phi^T*A^{-1}*Phi = Phi^T*(R^T*R)^{-1}*Phi
        #                  = Phi^T*R^{-1}*R^{-T}*Phi
        # v = R^{-t}*Phi
        # R^T*v = Phi --> get v

        y_pred = self.weight.T.dot(self.Phi)
        v = np.linalg.solve(self.R.T, self.Phi)
        var = self.sigma_n**2 * (1+v.T.dot(v))
        return y_pred, var


if __name__ == "__main__":
    f = lambda x: 5*np.exp(0.1*x)

    x = np.arange(-2.0, 2.0, 0.01)
    x = x.reshape(-1,1)

    x_test = np.arange(-3.0, 3.0, 0.01)
    x_test = x_test.reshape(-1,1)

    y = f(x)

    D = 100
    print(x.shape)
    hyper_param = np.array([1.0, 1.0, 0.01])
    iss_gpr = ISS_GPR(x,y,hyper_param,D)

    for i in range(len(x)):
        iss_gpr.update(x[i],y[i])

    y_pred = np.zeros((len(x_test),1))
    var = np.zeros((len(x_test),1))
    y_gnd = f(x_test)
    print(y_gnd.shape)

    for i in range(len(x_test)):
        y_pred[i], var[i] = iss_gpr.predict(x_test[i])

    # cmap = plt.get_cmap('jet_r')
    # # color1 = cmap(0.8)
    # # color2 = cmap(0.5)

    plt.plot(x_test,y_pred,linewidth=2, label="prediction")
    plt.fill_between(x_test.flatten(),
                           (y_pred-1.96*np.sqrt(var)).flatten(),
                           (y_pred+1.96*np.sqrt(var)).flatten(),
                           alpha=0.5)

    plt.plot(x_test,y_gnd,'--',linewidth=4,label="ground truth")
    plt.legend()
    plt.show()
    # plt.savefig("iss_gpr_example_exp.png")