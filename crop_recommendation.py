import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("Crop_recommendation.csv")

print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nCrop Types:")
print(data["label"].unique())

print("\nNumber of Crop Types:")
print(data["label"].nunique())

print("\nCrop Distribution:")
print(data["label"].value_counts())

plt.figure(figsize=(12, 6))

sns.countplot(data=data, x="label")

plt.xticks(rotation=45)
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()

print("\nStatistical Summary:")
print(data.describe())

plt.figure(figsize=(10, 7))

correlation = data.select_dtypes(include="number").corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

X = data.drop("label", axis=1)
y = data["label"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

results = {}

trained_models = {}

for name, model in models.items():

    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy
    trained_models[name] = model

    print(f"{name}: {accuracy:.4f}")

all_results = results

print("\nFinal Model Comparison:")

for name, accuracy in all_results.items():
    print(f"{name}: {accuracy:.4f}")

best_model_name = max(
    all_results,
    key=all_results.get
)

best_model = trained_models[best_model_name]

best_accuracy = all_results[best_model_name]

print("\nBest Model:")
print(best_model_name)

print("\nBest Accuracy:")
print(best_accuracy)

if best_model_name in ["KNN", "SVM"]:
    y_pred = best_model.predict(X_test_scaled)
else:
    y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.bar(
    all_results.keys(),
    all_results.values()
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print("\n========================================")
print("       CROP RECOMMENDATION SYSTEM")
print("========================================")

N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall (mm): "))

user_data = pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=X.columns
)

if best_model_name in ["KNN", "SVM"]:
    user_data_scaled = scaler.transform(user_data)
    prediction = best_model.predict(user_data_scaled)
else:
    prediction = best_model.predict(user_data)

print("\n========================================")
print("           PREDICTION RESULT")
print("========================================")

print("Recommended Crop:", prediction[0])

print("========================================")
