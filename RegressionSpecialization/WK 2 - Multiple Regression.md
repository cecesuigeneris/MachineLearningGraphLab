# Multiple Regression

## Key Concepts:
* Simple/two/multiple linear features or polynomial regression
* Matrix operation
* Cost of D-dimensional curve

  > RSS(w) = sum(yi - h(xi)w)^2

* The gradient of the cost

  > ∇RSS(w) = -2H^T(y-HW)

## Implementing ML algorithms to build up Multiple Regression model using Gradient Descent

### Convert to Numpy Array
SFrames offers a number of benefits to users (especially when using Big Data and built-in graphlab functions), Numpy is a library that allows for direct (and optimized) matrix operations.

SFrame --> 2D numpy array (also called a matrix): use graphlab's built in **.to_dataframe()** to convert the SFrame into a Pandas dataframe. Then use Panda's **.as_matrix()** to convert the dataframe into a numpy matrix.

```python
import graphlab
sales = graphlab.SFrame('kc_house_data.gl/')
import numpy as np

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1            # this is how you add a **constant column (intercept)** to an SFrame
    features = ['constant'] + features     # this is how you combine two lists
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)
```

> Test the function above
```python
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')      # the [] around 'sqft_living' makes it a list
print example_features[0,:]          # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0] # and the corresponding output
>```

### Define predictions function based on feature matrix and weights
```python
def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
```

### Computing the derivative

Squared difference between the observed output and predicted output for a single point:
(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)^2
We have k features and a constant. 

So the derivative with respect to weight w[i]:
2*(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)* [feature_i] 
= 2*error*[feature_i]

```python
def feature_derivative(errors, feature):
    derivative = 2*np.dot(errors, feature)
    return(derivative)
```

### Gradient Descent
Given a starting point we **update** the **current weights** by moving in the **negative** gradient direction. Negative gradient is the direction of **decrease** and we're trying to **minimize a cost function**.

```python
from math import sqrt

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights)             # make sure it's a numpy array
    while not converged:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        gradient_sum_squares = 0                  # initialize the gradient sum of squares
        
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)):             # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i], compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares+=np.dot(derivative,derivative)       #gradient_sum_squares = gradient_sum_squares + np.dot(derivative,derivative) -- same results
            weights[i] = weights[i] - step_size*derivative            #weights = weights - step_size*derivative ## results slightly different
        # compute the square-root of the gradient sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
```

### Running the Gradient Descent as Simple Regression
```python
# split the data into training and test data first
train_data,test_data = sales.random_split(.8,seed=0)

# let's test out the gradient descent (use TRAIN data to calculate weights)
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

model_test_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
print model_test_weights

# compute predictions on TEST data
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
test_simple_predictions = predict_output(test_simple_feature_matrix, model_test_weights) 
print test_simple_predictions[0]

# compute RSS on TEST data
def get_residual_sum_of_squares(prediction, data, outcome):
    residuals = outcome - prediction
    RSS = np.dot(residuals, residuals).sum()
    return(RSS) 

rss_test = get_residual_sum_of_squares(test_simple_predictions, test_data, test_data['price'])
print rss_test
```

### Running the Gradient Descent as Multiple Regression
```python
model_features = ['sqft_living', 'sqft_living15'] 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

model_multireg_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print model_multireg_weights

(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_multireg_predictions = predict_output(test_feature_matrix, model_multireg_weights) 
print test_multireg_predictions[0]

rss_multi = get_residual_sum_of_squares(test_multireg_predictions, test_data, test_data['price'])
print rss_multi
```
