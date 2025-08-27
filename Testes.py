# -*- coding: utf-8 -*- 
import pandas as pd    
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score   

# Carregar o conjunto de dados de treino
df = pd.read_csv("treino.csv")
print(df.head())

# Separar as variáveis de entrada e a variável alvo
X = df.drop(columns=['id', 'target'])
y = df['target']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo MLP
modelo = MLPClassifier(hidden_layer_sizes=(200, 100), activation='tanh', max_iter=1000, learning_rate_init=0.001, random_state=42)

# Treinar o modelo
modelo.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = modelo.predict(X_test)

# Avaliar o modelo
precisao = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {precisao:.4f}')

# Exibir a matriz de confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Exibir o relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Carregar o conjunto de dados de teste
df_teste = pd.read_csv('teste.csv')

# Preparar os dados de teste (remover a coluna 'id')
X_teste = df_teste.drop(columns=['id'])

# Fazer previsões para o conjunto de teste
y_teste_pred = modelo.predict(X_teste)

# Criar o DataFrame com os resultados das previsões
resultado = pd.DataFrame({'id': df_teste['id'], 'target': y_teste_pred})

# Salvar as previsões em um arquivo CSV
resultado.to_csv('previsoes.csv', index=False)

df = pd.read_csv('previsoes.csv')
print(df.head())
print(df.tail())
print(df.info())