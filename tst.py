# Testing : EDL
import numpy as np
import utility as ut

def forward_edl(X, W1, W2, W3):
    Z1 = X @ W1
    A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid
    Z2 = A1 @ W2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid
    Z3 = A2 @ W3
    Y_pred = np.exp(Z3) / np.sum(np.exp(Z3), axis=1, keepdims=True)  # Softmax
    return np.argmax(Y_pred, axis=1)  # Clase predicha

def matriz_confusion_manual(clases_reales, clases_predichas, num_clases):
    # Inicializar matriz de confusión
    matriz = np.zeros((num_clases, num_clases), dtype=int)

    # Llenar la matriz de confusión
    for real, predicha in zip(clases_reales, clases_predichas):
        matriz[np.argmax(real, axis=0), predicha] += 1

    return matriz

def calcular_metricas(matriz_confusion):
    # Para dos clases, extraer TP, TN, FP, FN
    TP = matriz_confusion[0, 0]
    TN = matriz_confusion[1, 1]
    FP = matriz_confusion[1, 0]
    FN = matriz_confusion[0, 1]

    # Calcular métricas
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, f1_score, accuracy

def main():
        pathTest='test'	
        X,y=ut.loadData(pathTest)
        pesos=np.load("Pesos.npz")
        W1,W2,W3=pesos['W1'],pesos['W2'],pesos['W3']
        y_pred=forward_edl(X, W1, W2, W3)
        matriz_confusion = matriz_confusion_manual(y, y_pred, 2)
        print("Matriz de confusión:",matriz_confusion)
        precision, recall, f1_score, accuracy = calcular_metricas(matriz_confusion)
        print("precision:",precision)
        print("recall:",recall)
        print("f1_score:",f1_score)
        print("accuracy:",accuracy)
              

if __name__ == '__main__':   
	 main()
