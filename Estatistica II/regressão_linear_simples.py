import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('cars.csv')
base = base.drop(['Unnamed: 0'], axis = 1)

x = base.iloc[:, 1].values
x = x.reshape(-1, 1)
y = base.iloc[:, 0]. values
correlacao = np.corrcoef(x, y)

modelo = LinearRegression()
modelo.fit(x, y)

modelo.intercept_
modelo.coef_

plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color = 'red')

# distância 22 pés
modelo.intercept_ + modelo.coef_ * 22

modelo.predict([[22]])

modelo._residues

visualizador = ResidualsPlot(modelo)
visualizador.fit(x, y)
visualizador.poof()