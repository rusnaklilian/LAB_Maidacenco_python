# Importarea bibliotecilor necesare pentru analiza și vizualizarea datelor
import numpy as np  # Pentru lucrul cu array-uri și operații numerice
import pandas as pd  # Pentru manipularea datelor în tabele (DataFrame)
import matplotlib.pyplot as plt  # Pentru generarea diverselor tipuri de grafice
import seaborn as sns  # Pentru vizualizarea statistică a datelor, bazată pe matplotlib
from sklearn.model_selection import train_test_split  # Pentru împărțirea datelor în seturi de antrenament și test
from sklearn.metrics import accuracy_score  # Pentru calcularea acurateței

# Încărcarea datelor dintr-un fișier CSV într-un DataFrame
file_path = 'wine-quality-white-and-red.csv'  # Calea către fișierul cu date
df = pd.read_csv(file_path)  # Citirea datelor din fișierul CSV și salvarea lor într-un DataFrame

# Conversia tipului de vin în format numeric
df['type'] = df['type'].map({'red': 0, 'white': 1})  # 'red' devine 0, iar 'white' devine 1

# Analiza statistică pentru fiecare caracteristică în funcție de tipul de vin
statistical_analysis = df.groupby('type').agg(['mean', 'median', 'std', 'var', lambda x: x.max() - x.min()])
print("Analiza statistică:\n", statistical_analysis)

# Calcularea corelației absolute pentru fiecare coloană cu tipul de vin
correlation = df.corr()['type'].abs().sort_values(ascending=False)
print("\nCorelația absolută cu tipul de vin:\n", correlation)

# Step 7: Pregătirea datelor pentru clasificare
# Selectăm caracteristica cu cea mai mare corelație absolută cu tipul de vin
max_corr_feature = correlation.index[1]  # A doua caracteristică (prima fiind tipul de vin)
X = df[max_corr_feature].values.reshape(-1, 1)  # Extragem caracteristica și o facem un array 2D
y = df['type'].values  # Variabila țintă ('type')

# Împărțirea datelor în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Setarea unui prag pentru predicție
threshold = X_train.mean()  # Calculăm media caracteristicii în setul de antrenament
print(f"Pragul pentru predicție: {threshold:.2f}")

# Step 9: Realizarea predicțiilor
y_pred = []  # Creăm o listă goală pentru predicții
for x in X_test:  # Parcurgem fiecare valoare din setul de test
    if x[0] > threshold:  # Dacă valoarea depășește pragul
        y_pred.append(1)  # Predicția este vin alb (1)
    else:
        y_pred.append(0)  # Predicția este vin roșu (0)

# Step 10: Calcularea acurateței
accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratețea predicției: {accuracy:.2f}")

# Step 11: Vizualizarea distribuției caracteristicii
plt.figure(figsize=(10, 6))
sns.histplot(df[df['type'] == 0][max_corr_feature], color='red', label='Red Wine', kde=True)
sns.histplot(df[df['type'] == 1][max_corr_feature], color='blue', label='White Wine', kde=True)
plt.title(f"Distribuția {max_corr_feature} pentru vinurile roșii și albe")
plt.legend()
plt.show()

# Vizualizarea distribuției acidității volatile în funcție de tipul de vin
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='volatile acidity', hue='type', kde=True)
plt.title('Distribuția acidității volatile pentru vinurile roșii și albe')
plt.show()

# Crearea unui boxplot pentru distribuția conținutului de alcool în funcție de tipul de vin
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='type', y='alcohol')
plt.title('Distribuția conținutului de alcool pentru vinurile roșii și albe')
plt.xticks([0, 1], ['Red', 'White'])
plt.show()

# Crearea unei hărți de căldură pentru matricea de corelație între caracteristicile vinurilor
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Matricea de corelație a caracteristicilor vinurilor')
plt.show()
