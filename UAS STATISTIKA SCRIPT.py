import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
from scipy.stats import ttest_ind

# Membaca file dari excel dan menetapkan nomor responden sebagai index
df = pd.read_csv('/Users/Ali Hakim/dev/Statistika/respondent1.csv')


# Menambahkan kolom A_INS dengan value mean dari INS1:INS5
df['A_INS'] = (df['INS1'] + df['INS2'] + df['INS3'] + df['INS4'] + df['INS5']) / 5

# Menambahkan kolom A_STUD dengan value mean dari STUD1:STUD5
df['A_STUD'] = (df['STUD1'] + df['STUD2'] + df['STUD3'] + df['STUD4'] + df['STUD5']) / 5

# Menambahkan kolom A_ELU dengan value mean dari ELU1:ELU4
df['A_ELU'] = (df['ELU1'] + df['ELU2'] + df['ELU3'] + df['ELU4']) / 4

# Menambahkan kolom A_SUP dengan value mean dari SUP:SUP4
df['A_SUP'] = (df['SUP1'] + df['SUP2'] + df['SUP3'] + df['SUP4']) / 4

# Menambahkan kolom A_TECH dengan value mean dari ELU1:ELU5
df['A_TECH'] = (df['TECH1'] + df['TECH2'] + df['TECH3'] + df['TECH4'] + df['TECH5']) / 5

head = df.head()
print(head)
X = df[['A_INS','A_STUD','A_SUP','A_TECH']]
Y = df['A_ELU']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)



dfMale = df[df['Gender'] == 'Pria']
dfFemale = df[df['Gender'] == 'Wanita']

print(dfMale.head())
print(dfFemale.head())



ttest, pval = ttest_ind(dfMale['ELU3'], dfFemale['ELU3'])
print('t-hitung: %0.4f, p-value: %0.4f' % (ttest, pval))
if pval <= 0.10:
    print("null hypothesis rejected, jadi terdapat perbedaan dalam keefektivitasan pembelajaran e-Learning bagi pria dan wanita")
else:
    print("fail to reject null hypothesis, jadi tidak terdapat perbedaan dalam keefektivitasan pembelajaran e-Learning bagi pria dan wanita")



