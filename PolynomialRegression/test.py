import operator
import os
import io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def load_data_lists(data_dir,file_name):
    with open(os.path.join(data_dir,file_name),"r") as f:
        lines = f.readlines()
        line = [ls.strip('\n') for ls in lines] 
        # \t data uses TAB to seperate x & y columns
        strX = ([l.strip().split('\t')[0] for l in line])
        strY = ([l.strip().split('\t')[1] for l in line])
        x = np.array(strX, dtype=np.float32)
        y = np.array(strY, dtype=np.float32)
    return x, y

x, y = load_data_lists("PolynomialRegression","data.txt")

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

try:
    mode=int(input('Give Polynomial degree: '))
except ValueError:
    print ("Not a int number")

polynomial_features= PolynomialFeatures(degree=mode)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print("RMSE:", rmse)
print("Regression Score: ", r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()