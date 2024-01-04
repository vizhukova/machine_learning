import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../dm_office_sales.csv")
print(df.head())
print(df.info())

plt.figure(figsize=(12,8))
# sns.scatterplot(x='salary',y='sales',data=df, hue='division')
# sns.scatterplot(x='salary',y='sales',data=df,hue='work experience')
# sns.scatterplot(x='salary',y='sales',data=df,hue='work experience',palette='viridis')

# sns.rugplot(x = 'salary', data = df, height=0.5)

sns.set(style='darkgrid')
# sns.displot(data = df, x = 'salary', kde = True)
# sns.histplot(data = df, x = 'salary')
# sns.kdeplot(data = df, x = 'salary')
# sns.countplot(data = df, x = 'division')

df = pd.read_csv("../StudentsPerformance.csv")
print(df.head())

# sns.countplot(x='division',data=df)
# sns.countplot(x='level of education',data=df)
# sns.countplot(x='level of education',data=df,hue='training level')
# sns.countplot(x='level of education',data=df,hue='training level',palette='Set1')
# sns.countplot(x='level of education',data=df,hue='training level',palette='Paired')
# sns.barplot(x='level of education',y='salary',data=df,estimator=np.mean,ci='sd')
# sns.barplot(x='level of education',y='salary',data=df,estimator=np.mean,ci='sd',hue='division')
# sns.barplot(x='level of education',y='salary',data=df,estimator=np.mean,ci='sd',hue='division')
# plt.legend(bbox_to_anchor=(1.05, 1))

# sns.boxplot(x='parental level of education',y='math score',data=df)
# sns.boxplot(x='parental level of education',y='math score',data=df,hue='gender')
# # Optional move the legend outside
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# # NOTICE HOW WE HAVE TO SWITCH X AND Y FOR THE ORIENTATION TO MAKE SENSE!
# sns.boxplot(x='math score',y='parental level of education',data=df,orient='h')
# sns.boxplot(x='parental level of education',y='math score',data=df,hue='gender',width=0.3)
# sns.violinplot(x='parental level of education',y='math score',data=df)
# sns.violinplot(x='parental level of education',y='math score',data=df,hue='gender')
# sns.violinplot(x='parental level of education',y='math score',data=df,hue='gender',split=True)
# sns.violinplot(x='parental level of education',y='math score',data=df,inner=None)
# sns.violinplot(x='parental level of education',y='math score',data=df,inner='box')
# sns.violinplot(x='parental level of education',y='math score',data=df,inner='quartile')
# sns.violinplot(x='parental level of education',y='math score',data=df,inner='stick')
# # Simply switch the continuous variable to y and the categorical to x
# sns.violinplot(x='math score',y='parental level of education',data=df,)
# sns.violinplot(x='parental level of education',y='math score',data=df,bw=0.1)

# sns.swarmplot(x='math score',data=df)
# sns.swarmplot(x='math score',data=df,size=2)
# sns.swarmplot(x='math score',y='race/ethnicity',data=df,size=3)
# sns.swarmplot(x='race/ethnicity',y='math score',data=df,size=3)
# sns.swarmplot(x='race/ethnicity',y='math score',data=df,hue='gender')
# sns.swarmplot(x='race/ethnicity',y='math score',data=df,hue='gender',dodge=True)
# sns.boxenplot(x='math score',y='race/ethnicity',data=df)
# sns.boxenplot(x='race/ethnicity',y='math score',data=df)
# sns.boxenplot(x='race/ethnicity',y='math score',data=df,hue='gender')

# sns.jointplot(x='math score',y='reading score',data=df)
# sns.jointplot(x='math score',y='reading score',data=df,kind='hex')
# sns.jointplot(x='math score',y='reading score',data=df,kind='kde')
# sns.pairplot(df)
# sns.pairplot(df,hue='gender',palette='viridis')
# sns.pairplot(df,hue='gender',palette='viridis',corner=True)
sns.pairplot(df,hue='gender',palette='viridis',diag_kind='hist')

plt.show()
