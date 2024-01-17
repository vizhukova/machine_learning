import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./DATA/GlobalTemperatures.csv")

print(df.sort_values(by='dt', ascending=False).head())
print(df.info())
print(df.columns)

plt.figure()
sns.scatterplot(x='dt',y='LandAverageTemperature',data=df,hue='LandAverageTemperature',palette='Dark2')
plt.show()

