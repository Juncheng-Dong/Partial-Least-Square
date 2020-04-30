import numpy as np
from numpy.linalg import pinv
import pandas as pd
import math
np.set_printoptions(precision=4)

import warnings
warnings.filterwarnings("ignore")
from numba import jit

from numpy.linalg import norm

def read_data(path,y_names):
    if path.endswith('xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    
    Y = df[['conscity','price','symboling']]
    X=df[list(set(df.columns)-set(Y.columns))]
    
    Y = Y.to_numpy()
    X = X.to_numpy()

    return (X,Y)

class PLS():
    def __init__(self, n_components):
        self.n_components = n_components
        self.T = []
        self.W = []
        self.Q = []
        self.U = []
        self.P = []
        self.B = []
    
    def normalize(self,Data):
        '''func to centerize and normalize the dataset,dataset should be numpy array'''
        D = Data.copy()
        mean = D.mean(axis=0)
        D = D-mean
        std = D.std(axis=0)
        D = D/std

        return D
    
    @jit(cache=True)
    def fit(self,X,Y,tol=1e-6):
        E,F=self.normalize(X),self.normalize(Y)
        T = []
        W = []
        Q = []
        U = []
        P = []
        B = []
        for _ in range(X.shape[1]):
            index=np.random.choice(range(Y.shape[1]))
            u=Y[:,index]
            counter = 0
            while(True):
                w = E.T@u
                w = w/norm(w)
                t = E@w
                t = t/norm(t)
                q = F.T@t
                q = q/norm(q)
                u = F@q

                if counter==0:
                    tlast=t
                elif norm(tlast-t)<tol:
                    break
                else:
                    tlast=t

                counter=counter+1

            b = t.T@u
            p = E.T@t
            B.append(b)
            T.append(t)
            P.append(p)
            W.append(w)
            Q.append(q)
            U.append(u)
            E = E-t.reshape(-1,1)@p.reshape(1,-1)
            F = F-b*t.reshape(-1,1)@q.reshape(1,-1)
        
        self.T = T
        self.W = W
        self.Q = Q
        self.U = U
        self.P = P
        self.B = B
        
        return (T,P,W,Q,U,B)
    
    def predict(self,X,n_components=None):
        from numpy.linalg import pinv
        
        if n_components==None:
            n = self.n_components
        else:
            n = n_components
            
        P = np.array(self.P[:n]).T
        B = np.diag(self.B[:n])
        Q = np.array(self.Q[:n]).T

        BPLS = pinv(P.T)@B@Q.T
        self.BPLS = BPLS
        
        return self.normalize(X)@BPLS
    
    def get_b(self):
        
        return self.BPLS