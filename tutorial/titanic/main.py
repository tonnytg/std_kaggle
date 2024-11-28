import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar os dados
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")

# Selecionar features relevantes
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = train_data[features]
X_test = test_data[features]

# Converter valores categóricos (como 'Sex') para numéricos
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
y = train_data['Survived']

# Dividir os dados de treino para validação
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

# Fazer previsões e calcular acurácia
predictions = model.predict(X_valid)
accuracy = accuracy_score(y_valid, predictions)
print(f"Accuracy: {accuracy}")

# Fazer previsões no conjunto de teste
test_predictions = model.predict(X_test)

# Preparar submissão
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})
output.to_csv('submission.csv', index=False)
print("Submission saved!")

