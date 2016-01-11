# Regression Modeling

Key concepts:
  - Residual sum of squares (RSS)
  - Overfitting via training/test split
  - Regression ML block diagram

## Predict House Prices Tutorial
Codes below utilize graphlab create and python language.

#### Fire up graphlab create and load data
```python
import graphlab
sales = graphlab.SFrame('home_data.gl/')
```

#### Exploring data 
```python
#print and view in ipython notebook instead of in canvas
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price") 
```

#### Create and build a simple regression model (sqft_living, price)
```python
#80% for training set, 20% for test; seed=0 to ensure same result performed by different user machines
train_data, test_data = sales.random_split(.8,seed=0)  
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living']
```

#### Evaluate the model
```python
sqft_model.evaluate(test_data) 
print test_data['price'].mean()
```

#### Show prediction
```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(test_data['sqft_living'], test_data['price'], '.',   
         test_data['sqft_living'],sqft_model.predict(test_data),'-')
sqft_model.get('coefficients')
```

#### Explore other features
```python
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
sales[my_features].show()
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')
```

#### Build a regression model with more features
```python
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)
print my_features
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)
```

#### Apply learned models to predict
```python
house1 = sales[sales['id'] == '5309101200']
print house1['price']
print sqft_model.predict(house1)
print my_features_model.predict(house1)

# Adding new data into the SFrame
bill_gates = {'bedrooms':[8],
              'bathrooms': [25], 
              'sqft_living': [50000], 
              'sqft_lot': [225000], 
              'floors':[4], 
              'zipcode':['98039'],
              'condition':[10],
              'grade':[10],
              'waterfront': [1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}
#call the dictionary created above
print my_features_model.predict(graphlab.SFrame(bill_gates)) 
```

## Predict House Prices Assignment
#### 1. Selection and summary statistics
```python
zipcode_98039 = sales[sales['zipcode']== '98039']
print zipcode_98039['price'].mean()
```

#### 2. Filtering data
```python
sf = sales[(sales['sqft_living'] >= 2000) & (sales['sqft_living'] <= 4000)] #data previously have been loaded using sales = graphlab.SFrame('home_data.gl/')
sf.num_rows()
sales.num_rows()
```

#### 3. Building a regression model with more features
```python
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)
print my_features_model.evaluate(test_data)
print advanced_features_model.evaluate(test_data)
```

