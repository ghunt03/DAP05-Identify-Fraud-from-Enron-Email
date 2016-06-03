

import pickle
import pprint
import pandas
import numpy as np
import matplotlib.pyplot as plt
enron_data = pickle.load(open("final_project/final_project_dataset.pkl", "r"))



df = pandas.DataFrame.from_dict(enron_data, orient='index', dtype=np.float)


print "Number of people " + str(len(df))
print "Number of features " + str(len(df.columns))
print "People of Interest " + str(df.groupby('poi').size()[1])

dfTotalPayments = df.loc[:,['poi', 'total_payments']]
totalRows = float(len(dfTotalPayments))
poi_with_no_payments = dfTotalPayments[dfTotalPayments['poi'] == 1].isnull().sum()[1]
non_poi_with_no_payments = dfTotalPayments[dfTotalPayments['poi'] == 0].isnull().sum()[1]
print "Percentage of POI with no payment data: " + str(poi_with_no_payments / totalRows)
print "Percentage of people with no payment data: " + str((non_poi_with_no_payments + poi_with_no_payments)  / totalRows)


dfNull = df.isnull().sum()
dfNull.plot(kind="barh")
plt.xlabel("Count of NaN")
plt.show()


print "Summary of Salary / Bonus information"
print df.describe().loc[:,['salary','bonus']]

