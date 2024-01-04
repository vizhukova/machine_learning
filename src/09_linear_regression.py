import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DATA/Advertising.csv")
print('Incoming data: \n', df.head())

df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
print('Updated data with new column "total_spend": \n', df.head())

# Show rgw incoming data in plot
# plt.figure()
# sns.scatterplot(data = df, x = 'total_spend', y = 'sales')
# sns.regplot(data = df, x = 'total_spend', y = 'sales')
# plt.show()


X = df['total_spend'] # feature metrics
y = df['sales'] # labels / expected output

# y = mx + b
# y = B1x + B0
# Least squares polynomial fit; deg = 1 as we have only one feature
y_exp = np.polyfit(X, y, deg = 1)
print(y_exp)
# result =  [0.04868788 4.24302822]
# 0.04868788 - for B1 and 4.24302822 for B0
# So now for any x it can predict y(sales)
B1 = y_exp[0]
B0 = y_exp[1]

get_predicted_sale = lambda spend: B1 * spend + B0

potential_spend = np.linspace(0, 500, 100)
predicted_sales = get_predicted_sale(potential_spend)
print(predicted_sales)

# sns.regplot(data = df, x = 'total_spend', y = 'sales')
# plt.plot(potential_spend, predicted_sales, color = 'red')
# plt.show()

spend = 200
print(get_predicted_sale(spend))

# example of 3 polinomial degree
y_exp = np.polyfit(X, y, deg = 3)
# result [ 3.07615033e-07 -1.89392449e-04  8.20886302e-02  2.70495053e+00]
# y = mx + b
# y = B3x^2 + B2x^2 + B1x + B0
print(y_exp)
B_3 = y_exp[0]
B_2 = y_exp[1]
B_1 = y_exp[2]
B_0 = y_exp[3]

get_pred_sale = lambda spend: B_3 * spend ** 3 + B_2 * spend ** 2 + B_1 * spend + B_0
pot_spend = np.linspace(0, 500, 100)
pred_sales = get_pred_sale(pot_spend)
print(pred_sales)

sns.regplot(data = df, x = 'total_spend', y = 'sales')
plt.plot(pot_spend, pred_sales, color = 'red')
plt.show()