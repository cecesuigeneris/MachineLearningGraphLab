# Simple Linear Regression

## Key concepts:
  - Residual sum of squares (RSS)
  - Concave/convex functions, finding *max via hill climbing/min via hill descent*
> Finding the optimum --> **derivative = 0**
>> While not converged,
>>                 w(t+1) <-- w(t) ± η*(dg(w)/dw)               
>> Note: t - iteration, η - step size

  - Machine learning algorithm 
> Finding **min sum(RSS (w0,w1))** = (sum(yi - (w0+w1xi))^2 using **Gradient Descent** (**multidimensional** hill descent)
>> While not converged, 
>>                 w(t+1) <-- w(t) - η*∇gw(t)                              
>> Note: choosing step size (fixed or decreasing) and convergence criteria (for convex functions, optimum occurs when derivative = 0, but in practice, stop when derivative < ε -- a threshold error to be set)




## Implementing ML Algorithms into Linear Regression Model for GraphLab

#### Fire up graphlab create and load data
```python
import graphlab
sales = graphlab.SFrame('kc_house_data.gl/')
```

#### Split data into training and testing
```python
#seed=0 to achieve same results by users. In practice, you may let Graphlab Create pick a random seed for you

train_data, test_data = sales.random_split(.8, seed=0)
```

#### Create and build a simple regression model (sqft_living, price)
```python
#80% for training set, 20% for test; seed=0 to ensure same result performed by different user machines

train_data, test_data = sales.random_split(.8,seed=0)  
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living']
```

#### Build a generic simple linear regression function
```python
input_feature = train_data['sqft_living']
output = train_data['price']

def simple_linear_regression(input_feature, output):
      # Compute 4 terms to calculate w0, w1 (i.e. intercept and slope): sum of input_feature, sum of output, sum of the product of input_feature and output, and sum of the input_feature squared

      sum_input = input_feature.sum()
      sum_output = output.sum()
      sum_product = (input_feature*output).sum()
      sum_squared = (input_feature*input_feature).sum()

      # Use the above 4 terms and the formula mentioned in **Approach 1** to calculate w0, w1 (i.e. intercept and slope). Note, calculate slope first, and use slope to calculate intercept

      slope = (sum_product - (sum_output*sum_input)/output.size())/(sum_squared - (sum_input*sum_input)/output.size())
      intercept = sum_output/output.size()-slope*(sum_input/output.size())

      return (intercept, slope)
```

> We can test the simple_linear_regression function below:
>> ```python
test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)
>> ```

After making sure the function is working, we can build a regression model to predict price based on sqft_living.

```python
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)
```

#### Show prediction
```python
def get_regression_predictions(input_feature, intercept, slope):
    predicted_values = intercept + input_feature*slope
    return predicted_values

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)
```

#### RSS
```python
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    predictions = input_feature*slope+intercept
    residuals = output-predictions
    RSS = (residuals*residuals).sum()
    return(RSS)

rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)
```

#### Inverse regression prediction
```python
def inverse_regression_predictions(output, intercept, slope):
    estimated_feature = (output - intercept)/slope
    return estimated_feature

my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)
```

#### New model: estimate prices based on bedrooms
```python
bedroom_intercept, bedroom_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])
print "Intercept: " + str(bedroom_intercept)
print "Slope: " + str(bedroom_slope)
```

#### Compare two models
```python
# Compute RSS when using bedrooms on TEST data (this dataset wasn't involved in learning the model, will help us comparing which model is better)

rss_prices_on_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bedroom_intercept, bedroom_slope)
print 'The RSS of predicting Prices based on bedrooms is : ' + str(rss_prices_on_bedrooms)

# Compute RSS when using squarefeet on TEST data

rss_prices_on_sqft_test = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft_test)
```
