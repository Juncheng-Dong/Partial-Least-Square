
# STA 663 Final Project

In this final project, we replicate the paper "PARTIAL LEAST-SQUARES REGRESSION: A TUTORIAL", and write and test a new PLS package. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pls.

```bash
pip install pls
```
## Usage
This pacakge contains one class called PLS(the number of components needs to be set) and six methods: read_data, fit, predict, get_b, variance, and mse.


```python
import pls

regressor=pls.PLS(n_components) #specify the number of components

regressor.read_data('path','ynames') #input: data path, and column names for Y; output: prepared data (X and Y) for regression
regressor.fit(X, Y) # 
regressor.predict(X)# returns predicted Y given X 
regressor.get_b() # returns B_PLS matrix, which are estimators for Y(s)
regressor.variance() #return a vector of variance explained by each component
regressor.mse() #return the mean squared error of this model 

```
### fit(self, X, y)
Parameters:

X: traning data, {array-like} of shape (n_samples, n_features) 

Y: target values, {array-like} of shape (n_samples, n_targets) 

Returns:

self: returns an instance of self.


### predict(self, X)
Parameters:

X: traning data, {array-like} of shape (n_samples, n_features) 
 
Returns:  {array}, predicted values (n_samples)

self: returns an instance of self.


### get_b(self)
 
Returns:  {array-like},  beta coefficients for dependdent variables (n_samples, n_targets)


### variance(self)
 
Returns:  {array},   return a vector of variance explained by each component (n_component)


### mse(self)
 
Returns:  {float}, mean squared error


self: returns an instance of self.
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

