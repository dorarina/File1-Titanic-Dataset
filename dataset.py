import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset into variable file
file = pd.read_csv("data/Titanic-Dataset.csv", index_col=0)


#clear the dataset
file.shape
#removing all NaN in column "Cabin"
file['Cabin'] = file['Cabin'].fillna('Unknown')
file = file.dropna(subset=['Age'])

#check the file, and its first 5 rows for clarification
print(file)
print(file.Age.head())

#Barchart of survival per gender

values = file.groupby('Sex')['Survived'].sum()
categories = values.index.tolist()

plt.bar(categories, values, color = 'pink')
plt.title('Survival count per gender')
plt.xlabel('Gender')
plt.ylabel('Survived')

plt.show()

#Piechart of classdistribution

labels= ['1st Class', '2nd Class', '3rd Class']
sizes = file['Pclass'].value_counts(sort=False)

plt.pie(sizes, labels = labels, autopct = '%1.1f%%', startangle=90)
plt.title('Class Distribution')

plt.show()

#Histogram of age distribution

data = file['Age']
#declaring number of bins by square root rule
bin_count = int(np.sqrt(len(data)))
plt.hist(data, bins = bin_count, color='skyblue', edgecolor = 'black')
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.show()