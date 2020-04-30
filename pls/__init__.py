def first_package():
    print("First Package !!!")
    
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
    
    Y = df[y_names]
    X=df[list(set(df.columns)-set(Y.columns))]
    
    Y = Y.to_numpy()
    X = X.to_numpy()

    return (X,Y)


def normalize(Data):
    '''func to centerize and normalize the dataset,dataset should be numpy array'''
    D = Data.copy()
    mean = D.mean(axis=0)
    D = D-mean
    std = D.std(axis=0)
    D = D/std

    return D
    

@jit(['(double[:,:], double[:,:],double,int32)'],cache=True)
def _fit(X,Y,n_components,tol=1e-6):
    E,F=normalize(X),normalize(Y)
    ssx =np.sum(np.square(E))
    ssy =np.sum(np.square(F))
    T = []
    W = []
    Q = []
    U = []
    P = []
    B = []
    xvariance=[]
    yvariance=[]

    if X.shape[1] < n_components:
        raise Exception("dimension error, training data has less columns than given n_components")

    for i in range(X.shape[1]):
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


        xvariance.append(p.T@p/ssx)
        yvariance.append(b**2/ssy)
        E = E-t.reshape(-1,1)@p.reshape(1,-1)
        F = F-b*t.reshape(-1,1)@q.reshape(1,-1)

    return (T,W,Q,U,P,B,xvariance,yvariance)
    

class PLS():
    def __init__(self, n_components):
        self.n_components = n_components
        self.trained = False
        self.xvariance=[]
        self.yvariance=[]
        self.T = []
        self.W = []
        self.Q = []
        self.U = []
        self.P = []
        self.B = []
    
    
    def fit(self,X,Y,tol=1e-6):
        self.X=X
        self.Y=Y
        
        T,W,Q,U,P,B,xvariance,yvariance=_fit(X,Y,tol,self.n_components)
        
        self.T=T
        self.W=W
        self.Q=Q
        self.U=U
        self.P=P
        self.B=B
        self.xvariance = xvariance
        self.yvariance = yvariance
        self.trained = True
        
        
    def predict(self,X,n_components=None):
        
        if self.trained == False:
            raise Exception("train the model first")
        
        if X.shape[1] != self.X.shape[1]:
            raise Exception("data to predict does not have same dimension as data trained")
        
        
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
        
        return normalize(X)@BPLS
    
    def get_b(self):
        P = np.array(self.P[:self.n_components]).T
        B = np.diag(self.B[:self.n_components])
        Q = np.array(self.Q[:self.n_components]).T

        BPLS = pinv(P.T)@B@Q.T
        return BPLS
    
    def variance(self):
        if self.trained == False:
            raise Exception("model has to be fitted first")
        
        return self.xvariance,self.yvariance
    
    def mse(self,n=None):
        if self.trained == False:
            raise Exception("model has to be fitted first")
        
        if n==None:
            n = self.X.shape[1]
        error = self.predict(self.X,n)-normalize(self.Y)
        
        return np.sum(np.square(error))/(self.Y.shape[0]*self.Y.shape[1])

#warming the package
_=PLS(1)
_x = np.random.rand(4,4)
_y = np.random.rand(4,1)
_.fit(_x,_y)
del _,_x,_y