from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 
from RandomForest import RandomForest
import pandas as pd 
from sklearn.metrics import ( 
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_ml_kelulusan.csv")
print (" data kebaca boskuhh")
print ("jumlah data ", df.shape)


target = 'lulus_tepat_waktu_4tahun'

X = df.drop(columns=[
    target,
    'lama_studi_tahun',
    'semester_tercatat'
]).values

y = df[target].values

X = X.astype(float)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


clf = RandomForest(n_trees=10)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, prediction)
print(acc)

print("\nClassification Report:")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))
# Menghitung jumlah data kelulusan
jumlah = df['lulus_tepat_waktu_4tahun'].value_counts()

# Menentukan label
labels = ['Tidak Tepat Waktu', 'Tepat Waktu']

values = [
    jumlah.get(0, 0),
    jumlah.get(1, 0)
]

plt.figure(figsize=(8, 5))

bars = plt.bar(labels, values)

plt.title('Distribusi Kelulusan Mahasiswa')
plt.xlabel('Status Kelulusan')
plt.ylabel('Jumlah Mahasiswa')

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()