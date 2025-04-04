import pandas as pd
Sheet1=pd.read_excel('/Users/meggn/Desktop/Projects/Diabetes Health Disparity Analysis/cdc-diabetes-2018.xlsx', sheet_name=0)
Diabetes=Sheet1.drop(columns=['YEAR','COUNTY','STATE'])
Sheet2=pd.read_excel('/Users/meggn/Desktop/Projects/Diabetes Health Disparity Analysis/cdc-diabetes-2018.xlsx', sheet_name=1)
Obesity=Sheet2.drop(columns=['YEAR','COUNTY','STATE'])
Sheet3=pd.read_excel('/Users/meggn/Desktop/Projects/Diabetes Health Disparity Analysis/cdc-diabetes-2018.xlsx', sheet_name=2)
Inactivity=Sheet3.drop(columns=['YEAR','COUNTY','STATE'])
df=pd.merge(Diabetes, Obesity, how='inner', on='FIPS')
Merged=pd.merge(df, Inactivity, how='inner', on='FIPS')
Final=Merged.drop(columns='FIPS')
Final
X=Final.drop(columns='% DIABETIC')
X
Y=Final['% DIABETIC']
Y

import seaborn as sns # type: ignore
sns.pairplot(Final, hue="% DIABETIC", diag_kind="hist", height=3)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, Y_train)
c=lr.intercept_
print("Intercept =",c)
m=lr.coef_
print("Slopes =",m)
Y_pred_train = lr.predict(X_train)
Y_pred_train
Y_pred_test = lr.predict(X_test)
Y_pred_test
 
from sklearn.metrics import r2_score
train_score=r2_score(Y_train, Y_pred_train)
print('The R2 value for Training Data is: ',train_score)
Y_pred_test = lr.predict(X_test)
test_score=r2_score(Y_test, Y_pred_test)
print('The R2 value for Testing Data is: ',test_score)
 
import matplotlib.pyplot as plt # type: ignore
sns.regplot(x=Y_train, y=Y_pred_train,data=Final,ci=None)
#plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Values of % Diabetic')
plt.ylabel('Predicted Values of % Diabetic')
plt.title('Training Data of Diabetes')
plt.show()
 
sns.regplot(x=Y_test, y=Y_pred_test,data=Final,ci=None)
#plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Values of % Diabetic')
plt.ylabel('Predicted Values of % Diabetic')
plt.title('Testing Data of Diabetes')
plt.show()
print("Multiple Linear Regression mean R-squared value =",r2_score(Y_test, Y_pred_test))
import statsmodels.api as sm # type: ignore
# Perform multiple linear regression using statsmodels
model = sm.OLS(Y, X).fit()
 
plt.scatter(Y, model.fittedvalues, label='Data Points')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Regression Line')
 
plt.xlabel('Actual Values of % Diabetic')
plt.ylabel('Predicted Values of % Diabetic')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()
 
Final.corr()
Final.describe()
import numpy as np
from scipy.stats import skew, kurtosis
stdev_residuals = np.std(Y)
print(f"Standard Deviation of Residuals: {stdev_residuals}")
skewness_residuals = skew(Y)
print(f"Skewness of Residuals: {skewness_residuals}")
kurtosis_residuals = kurtosis(Y)
print(f"Kurtosis of Residuals: {kurtosis_residuals}")
 
from sklearn.model_selection import cross_val_score,KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=5)
scores = cross_val_score(lr, X, Y, cv=kfold)
print("5-fold cross-validation mean R-squared value:", np.mean(scores))
from sklearn.preprocessing import PolynomialFeatures
max_degree = 5
r2_values = []
for degree in range(1, max_degree + 1):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    lr.fit(X_train_poly, Y_train)
    Y_pred_poly = lr.predict(X_test_poly)
    r2_poly = r2_score(Y_test, Y_pred_poly)
    r2_values.append(r2_poly)
    print(f"R2 Value for Polynomial Regression (Degree {degree}): {r2_poly}")
 
plt.plot(degree, r2_values, marker='o')
plt.title('R Squared Values vs. Polynomial Degrees')
plt.xlabel('Polynomial Degrees')
plt.ylabel('R Squared Values')
plt.grid(True)
plt.show()
 
X1=Final.drop(columns=['% INACTIVE','% DIABETIC'])
X1
X2=Final.drop(columns=['% OBESE','% DIABETIC'])
X2
degrees = [1, 2, 3, 4, 5]
fig, axs = plt.subplots(1, len(degrees), figsize=(15, 5))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(np.hstack((X1, X2)))
    lr.fit(X_poly, Y)
    x1_range = np.linspace(0, 2, 100)
    x2_range = np.linspace(0, 2, 100)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))
    X_grid_poly = poly_features.transform(X_grid)
    Y_grid = lr.predict(X_grid_poly)
 
cs = axs[i].contourf(X1_grid, X2_grid, Y_grid.reshape(X1_grid.shape), cmap='viridis')
axs[i].scatter(X1, X2, c=Y.ravel(), cmap='coolwarm', marker='o', label='Original Data')
axs[i].set_xlabel('% OBESE')
axs[i].set_ylabel('% INACTIVE')
axs[i].set_title(f'Degree {degree} Polynomial Regression')
fig.colorbar(cs, ax=axs[i])
plt.tight_layout()
plt.show()
