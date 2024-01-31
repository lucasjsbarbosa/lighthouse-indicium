# [Lighthouse] Desafio Ciência de Dados 2024

## Descrição
Este projeto desenvolve um sistema de predição de preços utilizando um modelo de redes neurais. O modelo é treinado para prever preços com base em características fornecidas. O projeto utiliza Python e Keras para a criação do modelo, juntamente com várias outras bibliotecas para pré-processamento de dados e análise exploratória. Vários algoritmos de regressão foram testados, mas as redes neurais apresentaram um resultado superior levando o conjunto de métricas em questão

## Instalação
Para executar este projeto, é necessário instalar as dependências listadas no arquivo `requirements.txt`. Recomenda-se o uso de um ambiente virtual Python para evitar conflitos de dependência. Siga os passos abaixo para configurar o ambiente:

1. Clone o repositório do projeto para o seu computador local.
2. Crie um ambiente virtual Python:
    ```bash
    python -m venv venv
    ```
3. Ative o ambiente virtual:
    - No Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - No macOS e Linux:
      ```bash
      source venv/bin/activate
      ```
4. Instale as dependências usando o pip:
    ```bash
    pip install -r requirements.txt
    ```

## Uso
Este projeto inclui scripts para treinar o modelo e fazer predições com dados novos.

### Preparando os Dados
Certifique-se de que seus dados estão formatados corretamente, seguindo o formato esperado pelo modelo. O script de predição espera dados no formato de um dicionário Python com características específicas.

### Fazendo Predições
Para fazer uma predição com o modelo treinado, execute o script `predict.py` (assumindo que você criou um script de predição baseado no código fornecido anteriormente). Se você estiver usando um Jupyter Notebook para fazer predições, siga as instruções contidas no notebook.

O script de predição automaticamente carrega os objetos `scaler`, `selector`, a lista de colunas de treinamento e o modelo treinado. Ele então aplica o mesmo pré-processamento aos novos dados antes de fazer a predição.

#### Exemplo de Código para Predição
```python
import pickle
from keras.models import load_model
import pandas as pd

# Carregar os objetos salvos
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('selector.pkl', 'rb') as f:
    selector = pickle.load(f)
with open('training_columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)
modelo = load_model('modelo_redes_neurais.h5')

# Novo dado para predição
novo_dado = {
    # Seu novo dado aqui
}

# Transformar e preparar novo dado
novo_dado_df = pd.DataFrame([novo_dado])
novo_dado_df = pd.get_dummies(novo_dado_df)
novo_dado_df = novo_dado_df.reindex(columns=training_columns, fill_value=0)
novo_dado_scaled = scaler.transform(novo_dado_df)
novo_dado_selected = selector.transform(novo_dado_scaled)

# Fazer a predição
predicao = modelo.predict(novo_dado_selected)
print("Predição de preço:", predicao[0][0])
