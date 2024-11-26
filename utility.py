
# My Utility : auxiliars functions
import numpy  as np

# Cargar datos encoder
def loadConfSae():
    conf=np.loadtxt('config_sae.csv').astype(int)
    return conf

# Cargar datos softmax
def loadConfSoft():
    conf=np.loadtxt('config_softmax.csv')
    return conf

# Cargar datos set
def loadData(path):
    data=np.loadtxt(f'd{path}.csv', delimiter=',')
    idx_igain=np.loadtxt('idx_igain.csv').astype(int)-1
    X=data[:,:-1]
    X=X[:,idx_igain]
    X=normData(X)
    Y=data[:,-1]
    y = np.array([[1, 0] if clase == 1 else [0, 1] for clase in Y])
    np.savetxt(f'Data{path.capitalize()}.csv',X,delimiter=',')
    return X,y

# Funcion para obtener w inicial
def getWInicial(d,L):
    r=np.sqrt(6/(d+L))
    return np.random.uniform(-r, r, size=(L, d))

# Funcion para calcular el ecm
def calcular_ecm(real, predicho):
    return np.mean((real - predicho) ** 2)

# Funcion Sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Funcion Normalizacion
def normData(X):
    epsilon = 1e-10
    muX = np.mean(X, axis=0)
    sigmaX = np.std(X, axis=0) 
    u = (X - muX) / (sigmaX + epsilon)
    return sigmoid(u)

# Inicializar pesos aleatorios
def inicializar_pesos(entradas, nodos_ocultos):
    r = np.sqrt(6 / (entradas + nodos_ocultos))
    return np.random.uniform(-r, r, size=(entradas, nodos_ocultos))

# Calcular pseudo-inversa
def calcular_pseudo_inversa(X, H, C):
    I = np.eye(H.shape[0])  # Matriz identidad
    A = H @ H.T + (I / C)  # Regularización
    return np.linalg.inv(A) @ H @ X.T

# Funcion SAE ELM
def sae_elm(X, Y, conf):
    capa_oculta_1 = conf[0]  # Número de nodos capa oculta 1
    capa_oculta_2 = conf[1]  # Número de nodos capa oculta 2
    C = conf[2]  # Factor de penalización
    runs = conf[3]  # Número de ejecuciones

    mejor_ecm = float('inf')
    mejor_W2 = None

    for run in range(runs):
        # Paso 1: Inicializar pesos aleatorios para la primera capa
        W1 = inicializar_pesos(X.shape[1], capa_oculta_1)
        H1 = sigmoid(X @ W1)  # Salida de la primera capa

        # Paso 2: Ajustar pesos con la pseudo-inversa
        W2 = calcular_pseudo_inversa(X.T, H1.T, C)
        H2 = sigmoid(H1 @ W2)  # Salida de la segunda capa

        # Paso 3: Ajustar pesos de salida con pseudo-inversa
        W3 = calcular_pseudo_inversa(Y.T, H2.T, C)  # W3.shape = (40, 2)

        # Paso 4: Calcular salida predicha y ECM
        Y_predicho = H2 @ W3
        ecm_actual = calcular_ecm(Y, Y_predicho)

        # Paso 5: Seleccionar la mejor matriz de pesos
        if ecm_actual < mejor_ecm:
            mejor_ecm = ecm_actual
            mejor_W2 = W2

        print(f"Run {run+1}/{runs}: ECM = {ecm_actual:.4f}")

    return W1,mejor_W2,W3

# Funcion Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Función para forward y backward pass
def forward_backward(X, Y, W1, W2, W3):
    Z1 = X @ W1
    A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid
    Z2 = A1 @ W2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid
    Z3 = A2 @ W3
    Y_pred = np.exp(Z3) / np.sum(np.exp(Z3), axis=1, keepdims=True)  # Softmax

    # Cálculo del costo
    costo = -np.mean(np.sum(Y * np.log(Y_pred + 1e-8), axis=1))

    # Gradientes
    dZ3 = Y_pred - Y
    dW3 = A2.T @ dZ3 / X.shape[0]
    dA2 = dZ3 @ W3.T * A2 * (1 - A2)
    dZ2 = dA2
    dW2 = A1.T @ dZ2 / X.shape[0]
    dA1 = dZ2 @ W2.T * A1 * (1 - A1)
    dZ1 = dA1
    dW1 = X.T @ dZ1 / X.shape[0]

    return costo, dW1, dW2, dW3

# Optimización con Adam por mini-batches
def adam_optimizer_batch(X, Y, W1, W2, W3, conf):
    max_epochs = conf[0].astype(int)
    batch_size = conf[1].astype(int)
    learning_rate = conf[2]
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-6

    # Inicialización de V y S
    V1, S1 = np.zeros_like(W1), np.zeros_like(W1)
    V2, S2 = np.zeros_like(W2), np.zeros_like(W2)
    V3, S3 = np.zeros_like(W3), np.zeros_like(W3)
    costos = []

    for epoch in range(max_epochs):
        # Paso 1: Reordenar aleatoriamente las muestras
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        Y = Y[indices]

        # Paso 2: Dividir en mini-batches
        for X_batch, Y_batch in crear_batches(X, Y, batch_size):
            # Forward y backward pass
            costo, dW1, dW2, dW3 = forward_backward(X_batch, Y_batch, W1, W2, W3)

            # Actualización de V y S para cada conjunto de pesos
            # Para W1
            V1 = beta1 * V1 + (1 - beta1) * dW1
            S1 = beta2 * S1 + (1 - beta2) * (dW1 ** 2)
            V1_corr = V1 / (1 - beta1 ** (epoch + 1))
            S1_corr = S1 / (1 - beta2 ** (epoch + 1))
            W1 -= learning_rate * V1_corr / (np.sqrt(S1_corr) + epsilon)

            # Para W2
            V2 = beta1 * V2 + (1 - beta1) * dW2
            S2 = beta2 * S2 + (1 - beta2) * (dW2 ** 2)
            V2_corr = V2 / (1 - beta1 ** (epoch + 1))
            S2_corr = S2 / (1 - beta2 ** (epoch + 1))
            W2 -= learning_rate * V2_corr / (np.sqrt(S2_corr) + epsilon)

            # Para W3
            V3 = beta1 * V3 + (1 - beta1) * dW3
            S3 = beta2 * S3 + (1 - beta2) * (dW3 ** 2)
            V3_corr = V3 / (1 - beta1 ** (epoch + 1))
            S3_corr = S3 / (1 - beta2 ** (epoch + 1))
            W3 -= learning_rate * V3_corr / (np.sqrt(S3_corr) + epsilon)

            costos.append(costo)

        # Mostrar progreso por época
        print(f"Época {epoch + 1}/{max_epochs}, Costo: {np.mean(costos[-len(X) // batch_size:]):f}")

    return W1, W2, W3, costos

# Función para crear mini-batches
def crear_batches(X, Y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]

# CResidual-Dispersion Entropy
def  label_binary(Y, Y_pred):
    return -np.mean(np.sum(Y * np.log(Y_pred + 1e-8), axis=1))
#
# CResidual-Permutation Entropy
def mtx_confusion():
    ...
    return
#

#

#

