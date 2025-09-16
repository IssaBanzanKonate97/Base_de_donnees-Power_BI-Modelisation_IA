#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from pathlib import Path
import glob, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sqlalchemy import create_engine
from getpass import getpass
from urllib.parse import quote_plus



# In[5]:


utilisateur = "root"
mot_de_passe  = getpass("MDP: ")       
hote = "localhost"
port = 3306              
base_de_donnee   = "ecommerce_2019"

data = create_engine(
    f"mysql+pymysql://{utilisateur}:{quote_plus(mot_de_passe)}@{hote}:{port}/{base_de_donnee}?charset=utf8mb4"
)


# In[8]:


from sqlalchemy import text
with data.begin() as connexion:
    connexion.execute(text("""
    CREATE OR REPLACE VIEW sales_2019 AS
        SELECT * FROM sales_january_2019
        UNION ALL SELECT * FROM sales_february_2019
        UNION ALL SELECT * FROM sales_march_2019
        UNION ALL SELECT * FROM sales_april_2019
        UNION ALL SELECT * FROM sales_may_2019
        UNION ALL SELECT * FROM sales_june_2019
        UNION ALL SELECT * FROM sales_july_2019
        UNION ALL SELECT * FROM sales_august_2019
        UNION ALL SELECT * FROM sales_september_2019
        UNION ALL SELECT * FROM sales_october_2019
        UNION ALL SELECT * FROM sales_november_2019
        UNION ALL SELECT * FROM sales_december_2019;
    """))
donnees = pd.read_sql("SELECT * FROM sales_2019", data)
donnees


# In[11]:


donnees.shape, donnees.head()


# In[12]:


donnees = donnees[['Order ID','Order Date','Quantity Ordered','Price Each','Purchase Address']]
donnees = donnees[donnees['Order Date'] != 'Order Date']
donnees['Order Date'] = pd.to_datetime(donnees['Order Date'], errors='coerce')
donnees['Quantity Ordered'] = pd.to_numeric(donnees['Quantity Ordered'], errors='coerce')
donnees = donnees.dropna(subset=['Order Date','Quantity Ordered']).copy()
donnees.head()


# In[13]:


quotidien = (
    donnees
      .assign(Jour = donnees['Order Date'].dt.floor('D'))
      .groupby('Jour', as_index=False)['Quantity Ordered'].sum()
      .rename(columns={'Quantity Ordered': 'y'})
      .sort_values('Jour')
)

quotidien.head(), quotidien.shape


# In[14]:


import numpy as np
quotidien['j'] = np.arange(len(quotidien))
X = quotidien[['j']]
y = quotidien['y']


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
X_train.shape, X_test.shape


# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")


# In[17]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
dates_test = quotidien['Jour'].iloc[-len(y_test):]
plt.plot(dates_test, y_test, label='Réel')
plt.plot(dates_test, y_pred, label='Prédict')
plt.title("prédire la quantité de commande par jou (régression linéaire)")
plt.legend()
plt.tight_layout()
plt.show()


# In[18]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = quotidien['j'].values.reshape(-1, 1)
y = quotidien['y'].values.reshape(-1, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)

reg = LinearRegression()
reg.fit(x_poly, y)

x_vals = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
x_vals_poly = poly_features.transform(x_vals)
y_vals = reg.predict(x_vals_poly)

plt.figure(figsize=(10,6))
plt.title("Une regression polynomiale", size=16)
plt.scatter(x, y, s=20)
plt.plot(x_vals, y_vals, c="red")
plt.show()


# In[ ]:





# In[ ]:




