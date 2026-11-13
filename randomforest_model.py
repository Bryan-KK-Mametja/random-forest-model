# importing modules
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# importing dataframe
df = pd.read_csv('sa_student_data.csv')
print(df.shape)

# cleaning df - dropping empty rows
df = df.dropna()

# dropping duplicates
df.drop_duplicates(inplace = True)

print(df.shape)

# splitting data into X and y - getting feature names
print(df.columns.tolist())
df = df.drop(columns = ['student_number', 'first_name', 'surname', 'module_code'])
print(df.columns.tolist())

# transforming object data types into int64
for i in df:
    if df[i].dtypes == object:
        df[i] = LabelEncoder().fit_transform(df[i])


X = df.loc[:, df.columns != 'grade_percent']
print(X)

y = df['grade_percent']
print(y)

# splitting into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# building model pipeline
model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(random_state = 42)
)

# fitting model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")


