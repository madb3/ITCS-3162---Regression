```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
```

### Introduction to the Data

##### This dataset shows travel between BOS (Boston), JFK (Queens, NY), LAS (Las Vegas), LAX (Los Angeles), ORD (Chicago), and SFO (San Francisco) airports recorded during June and July of 2024


```python
df = pd.read_csv("C:/Users/madb3/Downloads/flights.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Searched Date</th>
      <th>Departure Date</th>
      <th>Arrival Date</th>
      <th>Flight Lands Next Day</th>
      <th>Departure Airport</th>
      <th>Arrival Airport</th>
      <th>Number Of Stops</th>
      <th>Route</th>
      <th>Airline</th>
      <th>Cabin</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-06-30</td>
      <td>2024-07-16 23:48:00</td>
      <td>2024-07-17 15:03:00</td>
      <td>1</td>
      <td>LAS</td>
      <td>BOS</td>
      <td>1</td>
      <td>ATL</td>
      <td>Spirit Airlines</td>
      <td>Economy</td>
      <td>$83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-06-30</td>
      <td>2024-07-16 09:34:00</td>
      <td>2024-07-16 19:43:00</td>
      <td>0</td>
      <td>LAS</td>
      <td>BOS</td>
      <td>1</td>
      <td>EWR</td>
      <td>Spirit Airlines</td>
      <td>Economy</td>
      <td>$100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-06-30</td>
      <td>2024-07-16 23:48:00</td>
      <td>2024-07-17 15:03:00</td>
      <td>1</td>
      <td>LAS</td>
      <td>BOS</td>
      <td>1</td>
      <td>ATL</td>
      <td>Spirit Airlines</td>
      <td>Economy</td>
      <td>$78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-06-30</td>
      <td>2024-07-16 08:30:00</td>
      <td>2024-07-16 19:37:00</td>
      <td>0</td>
      <td>LAS</td>
      <td>BOS</td>
      <td>1</td>
      <td>IAH</td>
      <td>Spirit Airlines</td>
      <td>Economy</td>
      <td>$100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-06-30</td>
      <td>2024-07-16 07:00:00</td>
      <td>2024-07-16 16:36:00</td>
      <td>0</td>
      <td>LAS</td>
      <td>BOS</td>
      <td>1</td>
      <td>ORD</td>
      <td>American Airlines</td>
      <td>Basic Economy</td>
      <td>$148</td>
    </tr>
  </tbody>
</table>
</div>




```python
#See what data types we are working with
df.dtypes
```




    Searched Date            object
    Departure Date           object
    Arrival Date             object
    Flight Lands Next Day     int64
    Departure Airport        object
    Arrival Airport          object
    Number Of Stops           int64
    Route                    object
    Airline                  object
    Cabin                    object
    Price                    object
    dtype: object



### Data Cleaning

##### Drop irrelevant and redundant columns


```python
df = df.drop(columns = ['Searched Date','Route','Departure Date', 'Arrival Date'])
```


```python
set(df['Airline'].to_list())
```




    {'Alaska Airlines',
     'Allegiant Air',
     'Allegiant Air, Breeze Airways',
     'Allegiant Air, Spirit Airlines',
     'Allegiant Air, Sun Country Air',
     'American Airlines',
     'Avelo Airlines, Spirit Airlines',
     'Avianca',
     'Avianca, Spirit Airlines',
     'Breeze Airways, Spirit Airlines',
     'Cape Air',
     'Cape Air, United Airlines',
     'Delta',
     'Denver Air Connection, Sun Country Air',
     'Frontier',
     'Frontier, Spirit Airlines',
     'Hawaiian Airlines',
     'JetBlue',
     'Multiple airlines',
     'Porter Airlines',
     'Southern / Mokulele, Spirit Airlines',
     'Southern / Mokulele, Sun Country Air',
     'Spirit Airlines',
     'Spirit Airlines, Allegiant Air',
     'Spirit Airlines, Avelo Airlines',
     'Spirit Airlines, Avianca',
     'Spirit Airlines, Breeze Airways',
     'Spirit Airlines, Contour',
     'Spirit Airlines, Frontier',
     'Spirit Airlines, Southern / Mokulele',
     'Spirit Airlines, Sun Country Air',
     'Spirit Airlines, Volaris El Salvador',
     'Spirit Airlines, WestJet',
     'Sun Country Air',
     'Sun Country Air, Allegiant Air',
     'Sun Country Air, Porter Airlines',
     'Sun Country Air, Spirit Airlines',
     'Sun Country Air, WestJet',
     'TAP AIR PORTUGAL',
     'United Airlines',
     'United Airlines, Cape Air',
     'VivaAerobus',
     'Volaris',
     'Volaris El Salvador, Spirit Airlines',
     'WestJet',
     'WestJet, Sun Country Air'}



##### For the sake of ___, I want to hone in on major domestic airlines. I am only going to focus on JetBlue, UnitedAirlines, Spirit Airlines, Delta, American Airlines, Alaska Airlines, and Frontier.


```python
target_airlines = ['JetBlue', 'United Airlines', 'Spirit Airlines','Delta', 'American Airlines', 'Alaska Airlines', 'Frontier']
df_filtered = df[df['Airline'].apply(lambda x: any(airline in x for airline in target_airlines))]
print(df_filtered[['Airline', 'Price']])
```

                      Airline Price
    0         Spirit Airlines   $83
    1         Spirit Airlines  $100
    2         Spirit Airlines   $78
    3         Spirit Airlines  $100
    4       American Airlines  $148
    ...                   ...   ...
    317255    United Airlines  $219
    317256  American Airlines  $218
    317257    United Airlines  $198
    317258            JetBlue  $154
    317259    United Airlines  $228
    
    [314472 rows x 2 columns]
    


```python
set(df['Cabin'].to_list())
```




    {'Basic Economy',
     'BizFare',
     'Blue',
     'Blue Basic',
     'Blue Refundable',
     'Business/First',
     'Business/First (fully refundable)',
     'Comfort +',
     'Discount',
     'Economy',
     'Economy (fully refundable)',
     'Economy Plus',
     'First',
     'Main',
     'Main Cabin',
     'Main Cabin Basic',
     'Main Cabin Flex',
     'Main Select',
     'Mint',
     'Mixed',
     'No Flex Fare',
     'PorterClassic Basic',
     'Premium Economy',
     'Saver',
     'Standard',
     'UltraBasic'}



##### In order to focus on the most common cabin seats, we will only focus on Economy, Basic Economy, Blue, Blue Basic, Business/First, First, Main. We can consolodate some of these later.


```python
target_cabins = ['Economy', 'Basic Economy', 'Blue Basic','Business/First', 'First', 'Main']

df_filtered = df_filtered[df_filtered['Cabin'].apply(lambda x: any(cabin in x for cabin in target_cabins))]
print(df_filtered[['Cabin','Price']])
```

                    Cabin Price
    0             Economy   $83
    1             Economy  $100
    2             Economy   $78
    3             Economy  $100
    4       Basic Economy  $148
    ...               ...   ...
    317255  Basic Economy  $219
    317256  Basic Economy  $218
    317257        Economy  $198
    317258     Blue Basic  $154
    317259        Economy  $228
    
    [255186 rows x 2 columns]
    


```python

```

### Preprocessing

##### Converting price to a float to be able to train


```python
df["Price"] = df["Price"].replace(r"[\$,]", "", regex=True).astype(float)
```

##### Handling categorical features and converting them to numeric values


```python

```


```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Cabin_encoded'] = label_encoder.fit_transform(df['Cabin'])
df['Airline_encoded'] = label_encoder.fit_transform(df['Airline'])
```


```python
categorical_cols = ["Airline", "Cabin"]

label_encoders = {} 

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col]) 
    label_encoders[col] = le 


df = df.drop(columns=categorical_cols)
```


```python
#Visualize correlation using a heatmap
```


```python
df_numeric = df[['Number Of Stops', 'Price','Cabin_encoded','Airline_encoded','Flight Lands Next Day']]

correlation_matrix = df_numeric.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features and Price')
plt.show()
```


    
![png](output_23_0.png)
    


##### Number of stops has the highest correlation with price so far 


```python
#Scatterplot of key variables
```


```python

```

### Train Model


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
y = df["Price"]
X = df.drop(columns=["Price","Departure Airport","Arrival Airport"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#print("MAE:", mean_absolute_error(y_test, y_pred))
#print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

    Mean Squared Error: 38067.01741901176
    R² Score: 0.18519408219650935
    


```python
#Mean Squared Error of 38067 is a large margin of error
#The R2 Score of ~.1852 equals about 18.52%. This means that the model is only accurately _____ 18.52% of variants.
```


```python
### Combine ____ and ____ into one value - Distance
```


```python
#These values were found online and rounded to the nearest whole number
airport_distances = {
    ('LAS', 'BOS'): 2374,  
    ('LAS', 'LAX'): 236,   
    ('JFK', 'LAX'): 2469,  
    ('JFK', 'BOS'): 187,   
    ('LAX', 'BOS'): 2605,  
}

def get_distance(row):
    dep = row['Departure Airport']
    arr = row['Arrival Airport']
    
    if (dep, arr) in airport_distances:
        return airport_distances[(dep, arr)]
    elif (arr, dep) in airport_distances:
        return airport_distances[(arr, dep)]
    else:
        return None

df['Distance'] = df.apply(get_distance, axis=1)

print(df)
```

            Flight Lands Next Day Departure Airport Arrival Airport  \
    0                           1               LAS             BOS   
    1                           0               LAS             BOS   
    2                           1               LAS             BOS   
    3                           0               LAS             BOS   
    4                           0               LAS             BOS   
    ...                       ...               ...             ...   
    317255                      0               LAS             BOS   
    317256                      0               LAS             BOS   
    317257                      1               LAS             BOS   
    317258                      1               LAS             BOS   
    317259                      0               LAS             BOS   
    
            Number Of Stops  Price  Cabin_encoded  Airline_encoded  Distance  
    0                     1   83.0              9               22    2374.0  
    1                     1  100.0              9               22    2374.0  
    2                     1   78.0              9               22    2374.0  
    3                     1  100.0              9               22    2374.0  
    4                     1  148.0              0                5    2374.0  
    ...                 ...    ...            ...              ...       ...  
    317255                1  219.0              0               39    2374.0  
    317256                1  218.0              0                5    2374.0  
    317257                1  198.0              9               39    2374.0  
    317258                1  154.0              3               17    2374.0  
    317259                1  228.0              9               39    2374.0  
    
    [317260 rows x 8 columns]
    


```python
#Distance is a numeric value so it doesn't need to be encoded
```

#### Experiment: Linear Regression Model 


```python
print(df.columns)
```

    Index(['Flight Lands Next Day', 'Departure Airport', 'Arrival Airport',
           'Number Of Stops', 'Price', 'Cabin_encoded', 'Airline_encoded',
           'Distance'],
          dtype='object')
    


```python
#In order to get rid of NaN errors, I had to remove rows with NaN (null) values
df['Distance'] = df['Distance'].fillna(df['Distance'].mean())
```


```python
y = df["Price"]
X = df.drop(columns=["Price","Departure Airport","Arrival Airport"]) # Now all columns should be numeric

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on validation set
y_pred = lr_model.predict(X_val)

# Evaluate the model
r2 = r2_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"R-squared: {r2:.4f}")
print(f"RMSE (log dollars): {rmse:.4f}")

# Display coefficients for interpretation
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
})
print("Model Coefficients:")
display(coefficients)
```

    R-squared: 0.1893
    RMSE (log dollars): 194.6119
    Model Coefficients:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Flight Lands Next Day</td>
      <td>-22.854363</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Number Of Stops</td>
      <td>99.928369</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cabin_encoded</td>
      <td>7.404360</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airline_encoded</td>
      <td>0.795518</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Distance</td>
      <td>0.037687</td>
    </tr>
  </tbody>
</table>
</div>


### Model Interpretation

##### Flight lands next day: Flights that land the next day surprisingly have a $20 lower cost

##### Airline: A slight increase in price for certain airlines. This is tough to interpret because of encoding

##### Number of stops: Each stop increases price by $105.17

##### Cabin: A higher cabin slightly increases price


```python

```

##### Flight Lands next day: Flights that land the next day are $22 cheaper

##### Airline: Different airlines have different pricing, but airline is of low importance when determining price

##### Number of Stops: Each stop increases price by a little under $100 

##### Cabin: A higher class cabin still slightly increases price

##### Distance: Longer flights slightly increase ticket price

#### Insights

##### 1. Number of stops has the biggest impact on price so far

##### 2. Price may be driven more by route and time planned in advance

##### 3. Could explore more key factors missing from our data set: Booking time, time of the week?

### Positive Impacts

##### 1. Knowledge of price fluctations (and what causes them) can help people make the most cost-effective choices when booking flights

##### 2. Reveal patterns in flight pricing based on cabin, flight duration, and number of stops in between departure and destination

### Negative Impacts

##### 1. Airfare is updated in real time so it is hard to capture an accurate prediction because of unexpected events and other external factors that could hike or lower prices

##### 2. Airfare may not be linear as we've seen using this dataset. It may be based on entirely different features or a combination of features that weren't captured in this dataset.

### Conclusion

#### The data table provided a few features I thought would be important when determining flight price: Airline, Number of Stops (Layovers), Flight lands next day (Overnight flights), Cabin, and Flight Duration.

#### I thought airline and duration would be the most important features based on my own experience, but I was surprised to learn that number of stops (layovers) and overnight flights affected price the most. 

#### One of the main issues I ran into was encoding my categorical values into usable numeric values for the regression model. I could not use (One-Stop) binary encoding because there were more than 2 choices so I had to use label encoding, which may have skewed results a bit. 

#### Overall, this model could be improved by using a different dataset that included factors like economic conditions. A more comprehensive model could provide more accuracy and insight into flight pricing. Additionally, flight price may not be the best for linear regression because of how quickly it fluctuates based on many different factors.

### References

##### Flightconnections: All flights worldwide on a map! (n.d.-a). https://www.flightconnections.com/ 


```python

```
