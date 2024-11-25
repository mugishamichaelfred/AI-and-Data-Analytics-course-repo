import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt              
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({

    'X': np.arange(1,11),
    'y': [1.2, 2.3, 2.8, 4.4, 5.1, 6.3, 6.9, 8.2, 8.9, 10.3]
})
x = data[['X']]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, model.predict(x), color='red', label='Regression line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()