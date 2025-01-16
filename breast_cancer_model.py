import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class BreastCancerModel:
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self._initialize_pipeline()
        self._load_data()

    def _initialize_pipeline(self):
        """Inicia a pipeline com um escalador e o classificador RandomForest."""
        self.pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()), 
            ('model', RandomForestClassifier(random_state=self.random_state)),
        ])

    def _load_data(self):
        """Carrega o conjunto de dados e divide em treino e teste."""
        data = load_breast_cancer()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=self.test_size, random_state=self.random_state
        )
        self.classes = data.target_names

    def train(self):
        """Treina o modelo com os dados de treinamento."""
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self):
        """Faz previsões no conjunto de testes."""
        self.y_pred = self.pipeline.predict(self.X_test)
        self.y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

    def evaluate(self):
        """Exibe métricas e gráficos de avaliação."""
        self._plot_confusion_matrix()
        self._classification_report()
        self._plot_roc_curve()

    def _plot_confusion_matrix(self):
        """Exibe a matriz de confusão."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        plt.show()

    def _classification_report(self):
        """Exibe o relatório de classificação."""
        print("Relatório de Classificação:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.classes))

    def _plot_roc_curve(self):
        """Exibe a curva ROC."""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva de ROC')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    model = BreastCancerModel()
    model.train()
    model.predict()
    model.evaluate()
