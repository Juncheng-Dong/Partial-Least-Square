
# STA 663 Final Project

In this final project, we replicate the paper "PARTIAL LEAST-SQUARES REGRESSION: A TUTORIAL", and write and test a new PLS package. 

## required package
Numpy, Pandas, Numba

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pls.

```bash
pip install -i https://test.pypi.org/simple/ pls==1.0.1
```
## Usage
This pacakge contains one class called PLS(the number of components needs to be set) and five methods: fit, predict, get_b, variance, and mse.

All data passed into class methods are assumed to be numpy ndarray. Use read_data to prepare data . X,Y as numpy ndarray

```python
import pls

regressor=pls.PLS(n_components) #specify the number of components

regressor.read_data('path',['y1','y2']) #input: data path, and column names for Y; output: prepared data (X and Y) for regression
regressor.fit(X, Y) # 
regressor.predict(X)# returns predicted Y given X 
regressor.get_b() # returns B_PLS matrix, which are estimators for Y(s)
regressor.variance() #return a vector of variance explained by each component
regressor.mse() #return the mean squared error of this model 

```

## Methods
### 1. fit(self, X, y)

Parameters:

X: traning data, {array-like} of shape (n_samples, n_features) 

Y: target values, {array-like} of shape (n_samples, n_targets) 

Returns:

self: returns an instance of self.


### 2. predict(self, X)

Parameters:

X: traning data, {array-like} of shape (n_samples, n_features) 
 
Returns:  {array}, predicted values (n_samples)

self: returns an instance of self.


### 3. get_b(self)
 
Returns:  {array-like},  beta coefficients for dependdent variables (n_samples, n_targets)


### 4. variance(self)
 
Returns:  {array},   return a vector of variance explained by each component (n_component)


### 5. mse(self,n=self.n_components)
 
Returns:  {float}, mean squared error when using n components to predict. Default is using all components




## Example

Here we provide a simple example to walk through all methods in this package.

First, we need to import numpy and pls packages
```python
import pls
import numpy as np
```

Then, we have prepared two simulated X and Y in nd array format as our featrure data and response data
```python
#prepare X and Y in np array format 
X = np.array([
    [7,7,13,7],
    [4,3,14,7],
    [10,5,12,5],
    [16,7,11,3],
    [13,3,10,3]
])

Y=np.array(
[
    [14,7,8],
    [10,7,6],
    [8,5,5],
    [2,4,7],
    [6,2,4]
])
```

Now we call the pls class as regressor and set our number of principal components to use is 4 

```python
#pre set the number of components as 4
regressor = pls.PLS(4)
```
We feed our training data X and Y into this PLS model

```python
#fit the model
regressor.fit(X,Y)
```

We can check the mean squared error, which is a scale to show the fitness of this model.

```python
#set the number of principal components as 4
print("Mean Squared Error is:" , regressor.mse(4))
```

    Mean Squared Error is: 0.04166666666666666


We can also get how much variances of X and Y are explained by each component.


```python
#get X and Y variances that are expained by PLS
xv,yv=regressor.variance()
```


```python
print(" X variance explained:\n\n", xv)
```

     X variance explained:
    
     [0.7045058315507844, 0.2789576859794002, 0.01653648246981553, 2.5123601249280894e-32]



```python
print(" Y variance explained:\n\n", yv)
```

     Y variance explained:
    
     [0.6333236475046795, 0.22063802681243821, 0.1043716590162159, 0.0007355154907388358]



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

