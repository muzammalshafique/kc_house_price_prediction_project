#Question 1
import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Display the data types of each column
print(df.dtypes)

# Question 2
import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Drop the "id" column from the DataFrame
df.drop("id", axis=1, inplace=True)

# Print out a statistical summary of the remaining columns
print(df.describe())

# Question 3
import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Use the value_counts() method to count the number of houses with unique floor values
floor_counts = df['floors'].value_counts()

# Convert the resulting Series to a DataFrame
floor_counts_df = floor_counts.to_frame()

# Print out the resulting DataFrame
print(floor_counts_df)

#Question 4
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Use the boxplot() function to compare the distribution of prices for houses with and without a waterfront view
sns.boxplot(x='waterfront', y='price', data=df)

# Display the plot
# plt.show()

#Question 5
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Use the regplot() function to visualize the correlation between sqft_above and price
sns.regplot(x='sqft_above', y='price', data=df)

# Display the plot
# plt.show()

#Question 6
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['sqft_living']], df['price'], test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
reg = LinearRegression().fit(X_train, y_train)

# Calculate the R^2 on the test data
r_squared = reg.score(X_test, y_test)

# Print the R^2
print("R^2:", r_squared)


#Question 7
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Define the list of features
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
df[features] = df[features].dropna()
df['price'] = df['price'].dropna()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df['price'], test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
reg = LinearRegression().fit(X_train, y_train)

# Calculate the R^2 on the test data
r_squared = reg.score(X_test, y_test)

# Print the R^2
print("R^2:", r_squared)

#question 8
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the dataset into a pandas DataFrame
df = pd.read_csv('kc_house_data.csv')

# Define the features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
target = "price"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create a pipeline object
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly_transform", PolynomialFeatures(degree=2)),
    ("linear_regression", LinearRegression())
])

# Fit the pipeline object to the training data
pipeline.fit(X_train, y_train)

# Calculate the R^2 on the test data
r_squared = pipeline.score(X_test, y_test)

# Print the R^2
print("R^2:", r_squared)

#question 9
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# create polynomial features object
poly = PolynomialFeatures(degree=2)

# fit and transform the training data using the polynomial features
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# create Ridge regression object
ridge = Ridge(alpha=0.1)

# fit the model using the training data
ridge.fit(X_train_poly, y_train)

# calculate the R^2 using the test data
ridge_score = ridge.score(X_test_poly, y_test)
print('Ridge regression R^2 score:', ridge_score)

#Question 10
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Perform second order polynomial transform on training and testing data
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

# Fit a Ridge regression object with regularization parameter set to 0.1
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X_train_poly, y_train)

# Calculate R^2 score using test data
y_test_pred = ridge_reg.predict(X_test_poly)
r2 = r2_score(y_test, y_test_pred)

# Print R^2 score
print('R^2 score:', r2)