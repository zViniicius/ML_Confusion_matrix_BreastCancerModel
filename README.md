# Breast Cancer Classification with Random Forest

Este projeto utiliza o conjunto de dados de câncer de mama (`load_breast_cancer`) do `scikit-learn` para treinar um modelo de classificação utilizando a técnica de Random Forest. A pipeline inclui a normalização dos dados e a avaliação do modelo com a geração de métricas como a Matriz de Confusão, Relatório de Classificação e Curva ROC.

## Requisitos

- Python 3.x
- Bibliotecas necessárias:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

Você pode instalar as dependências usando o `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Como Usar

1. Clone o repositório ou baixe os arquivos do projeto.
2. Execute o script `breast_cancer_model.py` para treinar o modelo e avaliar os resultados.

```bash
python breast_cancer_model.py
```

### Descrição do Código

O projeto é estruturado dentro de uma classe `BreastCancerModel`, que encapsula a lógica de treinamento e avaliação do modelo. A classe possui os seguintes métodos:

- **`__init__`**: Inicializa o modelo e carrega os dados de câncer de mama.
- **`train`**: Treina o modelo usando Random Forest.
- **`predict`**: Faz previsões no conjunto de teste.
- **`evaluate`**: Exibe as métricas de avaliação, incluindo a Matriz de Confusão, Relatório de Classificação e Curva ROC.

### Métodos de Avaliação

- **Matriz de Confusão**: Exibe a matriz de confusão, mostrando a quantidade de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
- **Relatório de Classificação**: Mostra as métricas de precisão, recall, f1-score e suporte para cada classe.
- **Curva ROC**: Exibe a curva ROC e calcula a área sob a curva (AUC).

### Exemplo de Saída

Ao executar o script, você verá o seguinte tipo de saída:

- **Matriz de Confusão**: Visualização gráfica da matriz de confusão.
- **Relatório de Classificação**: Métricas como precisão, recall e F1-score para cada classe.
- **Curva ROC**: Gráfico da curva ROC com a área sob a curva (AUC).

### Exemplo de Relatório de Classificação

```plaintext
Relatório de Classificação:
              precision    recall  f1-score   support

   malignant       0.98      0.97      0.97       130
      benign       0.96      0.98      0.97       139

    accuracy                           0.97       269
   macro avg       0.97      0.97      0.97       269
weighted avg       0.97      0.97      0.97       269
```

### Contribuição

Sinta-se à vontade para fazer contribuições para melhorar este projeto. Se você tiver algum problema ou sugestão, abra um **issue** ou envie um **pull request**.
