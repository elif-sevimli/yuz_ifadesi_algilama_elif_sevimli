import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# CSV'den veriyi oku
df = pd.read_csv("veriseti.csv")

# X: Landmark verileri, y: Etiket (ifade)
X = df.drop("ifade", axis=1)
y = df["ifade"]

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test verisiyle doğruluk hesapla
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Doğruluk Oranı: {accuracy * 100:.2f}%")

# Modeli kaydet
joblib.dump(model, "model.pkl")
print("Model 'model.pkl' olarak kaydedildi.")
