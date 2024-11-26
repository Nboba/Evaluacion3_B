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


def main():
        #Carga e Inicializacion de datos
        pathTest='test'	
        X,y=ut.loadData(pathTest)
        pesos=np.load("Pesos.npz")
        W1,W2,W3=pesos['W1'],pesos['W2'],pesos['W3']
        y_pred=forward_edl(X, W1, W2, W3)

        #Matriz de Confusion
        matriz_confusion = ut.mtx_confusion(y, y_pred, 2)
        print(f"Matriz de confusión:\n{matriz_confusion}")
        
        #Metricas
        f1_score_1,f1_score_2 = ut.calcular_metricas(matriz_confusion)
        print("f1_score clase 1:",f1_score_1)
        print("f1_score clase 2:",f1_score_2)

        #Guardar archivos
        np.savetxt("confusion.csv", matriz_confusion, delimiter=",", fmt="%d", comments="")
        with open("fscores.csv", "w") as file:
            file.write(f"{f1_score_1}\n{f1_score_2}\n")
        
              

if __name__ == '__main__':   
	 main()
