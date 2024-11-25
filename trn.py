# Extreme Deep Learning
import numpy      as np
import utility    as ut

def train_edl(pathTrain):    
    #Cargar datos
    X,y=ut.loadData(pathTrain)    
    confSae=ut.loadConfSae()
    ut.sae_elm(X, y, confSae)   #Esto no esta haciendo nada
    confAdam=ut.loadConfSoft()

    #Calcular weights
    w1,w2,w3=ut.sae_elm(X, y, confSae)
    W1,W2,W3, costos=ut.adam_optimizer_batch(X, y,w1,w2,w3,confAdam)

    #Guardar archivos
    np.savez("Pesos.npz", W1=W1, W2=W2, W3=W3)
    np.savetxt("costo.csv", costos, delimiter=",", header="Costo", comments="")

# Beginning ...
def main(): 
    pathTrain='train'
    train_edl(pathTrain)           
    
       
if __name__ == '__main__':   
	 main()

