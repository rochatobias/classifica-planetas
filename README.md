# Projeto de Classificação de Planetas com Redes Neurais

Este projeto utiliza um modelo de Machine Learning, especificamente uma Rede Neural Artificial (Multi-layer Perceptron), para classificar dados, relacionados a sinais astronômicos para a detecção de planetas. O objetivo é treinar um modelo capaz de prever uma variável alvo (target) com base em um grande conjunto de características numéricas não definidas.

### 🎯 Escopo do Projeto
O script principal (`Testes.py`) realiza um fluxo completo de Machine Learning:

- Carrega um conjunto de dados de treino (`treino.csv`).

- Prepara os dados, separando as variáveis preditoras (features) da variável alvo (target).
- Divide os dados de treino em um conjunto de treino e um de validação para avaliar a performance do modelo internamente.
- Cria e treina um modelo `MLPClassifier` da biblioteca `scikit-learn`.
- Avalia o modelo treinado usando a acurácia e a matriz de confusão no conjunto de validação.
- Realiza previsões em um novo conjunto de dados (`teste.csv`) que não possui a variável alvo.
- Salva as previsões em um arquivo de submissão (`Previsoes.csv`).

### 📁 Estrutura dos Arquivos
- **treino.csv**: Contém os dados para treinamento e validação do modelo. Inclui as features, um id único e a coluna target que queremos prever.

- **teste.csv**: Contém os dados que devem ser classificados pelo modelo treinado. Possui a mesma estrutura do treino.csv, mas sem a coluna target.
- **Testes.py**: O script Python que executa todas as etapas do projeto, desde o treinamento até a geração das previsões.
- **Previsoes.csv**: O arquivo de saída gerado pelo script, contendo o id de cada amostra do conjunto de teste e a respectiva predição da target.

### 🤖 Modelo Utilizado
Foi utilizada uma Rede Neural do tipo Multi-layer Perceptron Classifier (MLPClassifier). A arquitetura definida no script é:

- Camadas Ocultas: Duas camadas, a primeira com 200 neurônios e a segunda com 100 neurônios (`hidden_layer_sizes=(200, 100)`).
- Função de Ativação: Tangente Hiperbólica (`activation='tanh'`).

### ⚙️ Como Executar o Projeto
#### 1. Pré-requisitos
Certifique-se de ter o Python 3 instalado. Você precisará das seguintes bibliotecas:

- `pandas`

- `scikit-learn`

#### 2. Instalação das Dependências
Abra o terminal ou prompt de comando e instale as bibliotecas necessárias:

```
pip install pandas scikit-learn
```

#### 3. Execução do Script
Com as dependências instaladas e todos os arquivos (Testes.py, treino.csv, teste.csv) na mesma pasta, execute o seguinte comando no terminal:

```
python Testes.py
```

#### 4. Resultados
Após a execução, o script irá:

1. Imprimir no console a acurácia do modelo no conjunto de validação.
2. Exibir a Matriz de Confusão para uma análise mais detalhada dos erros e acertos.
3. Gerar o arquivo Previsoes.csv no mesmo diretório, contendo as predições para os dados de teste.