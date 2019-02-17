import csv 
import numpy as np 
import matplotlib.pyplot as plt 
  
  
def loadCSV(filename): 
    ''' 
    function to load dataset 
    '''
    with open("C:/Users/User/Desktop/NEWDATA.csv","r") as csvfile: 
        lines = csv.reader(csvfile) 
        dataset = list(lines) 
        for i in range(len(dataset)): 
            dataset[i] = [x for x in dataset[i]]      
    return np.array(dataset) 
  
  
def normalize(X): 
    ''' 
    function to normalize feature matrix, X 
    '''
    #mins = np.amin(X, axis=None, out=None, keepdims=False)
    mins=300
    #maxs = np.max(X)  
    maxs=900
    rng = maxs - mins 
    #norm_X = 1 - ((maxs - X)/rng) 
    norm_X=60
    return norm_X 

  
  
def logistic_func(beta, X): 
    ''' 
    logistic(sigmoid) function 
    '''
    return 1.0/(1 + np.exp(-np.dot(X, beta.T))) 
  

import pandas as pd

print("training data")
ndf = pd.read_csv("C:/Users/User/Desktop/NEWDATA.csv")
df = pd.read_csv("C:/Users/User/Desktop/trainingdatatotal.csv")
print(ndf)


def log_gradient(beta, X, y): 
    ''' 
    logistic gradient function 
    '''
    first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1) 
    final_calc = np.dot(first_calc.T, X) 
    return final_calc 
  

def cost_func(beta, X, y): 
    ''' 
    cost function, J 
    '''
    log_func_v = logistic_func(beta, X) 
    y = np.squeeze(y) 
    step1 = y * np.log(log_func_v) 
    step2 = (1 - y) * np.log(1 - log_func_v) 
    final = -step1 - step2 
    return np.mean(final) 

tdf= pd.read_csv("C:/Users/User/Desktop/test data.csv")
print("\n Testing data \n", tdf) 
def grad_desc(X, y, beta, lr=.01, converge_change=.001): 
    ''' 
    gradient descent function 
    '''
    cost = cost_func(beta, X, y) 
    change_cost = 1
    num_iter = 1

    while(change_cost > converge_change): 
        old_cost = cost 
        beta = beta - (lr * log_gradient(beta, X, y)) 
        cost = cost_func(beta, X, y) 
        change_cost = old_cost - cost 
        num_iter += 1
      
    return beta, num_iter  

print("\n Resulting data \n")
print(df)
def pred_values(beta, X): 
    ''' 
    function to predict labels 
    '''
    pred_prob = logistic_func(beta, X) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value) 
  
  
def plot_reg(X, y, beta): 
    ''' 
    function to plot decision boundary 
    '''
    # labelled observations 
    x_0 = X[np.where(y == 0.0)] 
    x_1 = X[np.where(y == 1.0)] 
      
    # plotting points with diff color for diff label 
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0') 
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 
      
    # plotting decision boundary 
    x1 = np.arange(0, 1, 0.1) 
    x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2] 
    plt.plot(x1, x2, c='k', label='reg line') 

    
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.legend() 
    plt.show() 
      
import matplotlib.pyplot as pls 

col1=df['Customer_id']
col2= df['Credit Score']
pls.plot(col1,col2,marker='o', linestyle='--', label='loaded from file')
pls.xlabel('id')
pls.ylabel('score')
pls.legend()
pls.show()

print(b)
gdf= df.groupby('Customer_id')
ydf= gdf['Credit Score'].sum()
#print(ydf)
fgdf = gdf['Credit Score'].agg(np.mean)
fgdf.columns=['cus_id','score']
print(fgdf)

if __name__ == "__main__": 
    # load the dataset 
    dataset = loadCSV('dataset1.csv') 
      
    # normalizing feature matrix 
    X = normalize(dataset[:, :-1]) 
      
    # stacking columns wth all ones in feature matrix 
   # X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X)) 
  
    # response vector 
    y = dataset[:, -1] 
  
   
    # predicted labels 
    #y_pred = pred_values(beta, X) 
      
    # number of correctly predicted labels 
    #print("Correctly predicted labels:", np.sum(y == y_pred)) 
    
    # plotting regression line 
    #plot_reg(X, y, beta) 