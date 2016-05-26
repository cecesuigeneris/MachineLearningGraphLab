# Assessing Performance: Bias-Variance Tradeoff

## Key Concepts:
* Loss function (cost of using w hat at x when y is true): L(y, fw(x))
  > Absolute error: L(y, fw(x)) = |y - fw(x)|
  > Sqaured error: L(y, fw(x)) = (y - fw(x))^2
  
* Training error (average error of loss function)
  > e.g. use sqaured error as loss function, training error (w hat) = RMSE = ave.sum(yi - fw(x)^2
  > training error decreases with increased model complexity

* Generalization/true error
  > Ex,y[loss function] = average over all possible (x,y) pairs weighted by how likely each is
  > cannot compute because it is impossible to look at all possible dataset
  > error decreases and increases with increasing model complexity
  
* Test error (approx generalization error)
  > ave.sum[loss function] based on testset, however [loss function] fit fw(x) using training data
  > fluctuated along generalization error plot
  > overfitting if 1) training error w' > w hat, 2) true error w' < w hat

* Training set/test set split
* 3 sources of errors: noise (irreducible), bias (low complexity, high bias), variance (low complexity, low variance)
  > Bias-variance tradeoff: MSE (mean sqaured error) = bias^2 + variance <-- cannot compute
* Error vs amount of data (for a fixed model complexity)
  > in the limit, true error = training error
* Workflow for regression model
  1. model selection:
    > choose model complexity λ
      - estimate parameter wλ on training data
      - assess performance of wλ on test data
      - choose λ* to be λ with lowest test error
  2. model assessment
    > compute test error of wλ* to approximate generalization error
    
* Solution for model is overly optimistic (i.e. test data is not representative of the whole world)
    > split dataset into training set, validation set, and test set
      - select λ* such that wλ* minimizes errors on validation set
      - approximate generalization error of wλ* using test set


## Polynomial Regression 

### Define polynomial_sframe function

```python
import graphlab
sales = graphlab.SFrame('kc_house_data.gl/')

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature**power
    return poly_sframe
```

### Visualize polynomial regression
```python
# for plotting purpose (connecting the dots), sort by the values of sqft_living
sales = sales.sort(['sqft_living', 'price'])

# start with a degree 1 polynomial to predict price
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target
model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
model1.get("coefficients")

# use matplotlib to visualize graph
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')
        
# same steps for polynomial degree of 3
poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features = poly3_data.column_names() 
poly3_data['price'] = sales['price'] 
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, validation_set = None)
model3.get("coefficients")

plt.plot(poly3_data['power_3'],poly3_data['price'],'.',
        poly3_data['power_3'], model3.predict(poly3_data),'-')
```

### Splitting sales data into subsets

```python
# split into 4 equal size data set
(set_A, set_B) = sales.random_split(0.5, seed=0)
(set_1, set_2) = set_A.random_split(0.5, seed=0)
(set_3, set_4) = set_B.random_split(0.5, seed=0)

# fit a 15th degree polynomial
poly15_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_1['price']
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
plt.plot(poly15_data['power_15'],poly15_data['price'],'.',
        poly15_data['power_15'], model15.predict(poly15_data),'-')
```

### Selecting a polynomial degree
