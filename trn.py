# Extreme Deep Learning

import numpy      as np
import utility    as ut
import pandas     as pd



def train_edl(pathTrain):    
    X,y=ut.loadData(pathTrain)    
    confSae=ut.loadConfSae()
    ut.sae_elm(X, y, confSae)
    confAdam=ut.loadConfSoft()
    w1,w2,w3=ut.sae_elm(X, y, confSae)
    W1,W2,W3, costos=ut.adam_optimizer_batch(X, y,w1,w2,w3,confAdam)
    np.savez("Pesos.npz", W1=W1, W2=W2, W3=W3)
    pd.DataFrame(costos, columns=["Costo"]).to_csv("costo.csv", index=False)

# Beginning ...
def main(): 
    pathTrain='train'
    train_edl(pathTrain)           
    
       
if __name__ == '__main__':   
	 main()

