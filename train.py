import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("salary_data.csv", sep=';')
print(df.columns)

X = df[['YearsExperience']]  
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, model.predict(X_test))

print(f"Akurasi Model: {accuracy * 100:.2f}%")
print(f"MAE: ${mae:,.2f}")

res = model.predict([[1]])
print(f"Prediksi Gaji (1 Tahun Pengalaman): ${res[0]:,.2f}")

joblib.dump(model, "salary_prediction_model.pkl")
print("Model berhasil disimpan!")