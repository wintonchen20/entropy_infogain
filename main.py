import numpy as np

def entropy(Y):
    
    num_elements = Y.shape[0]
    
    num_true = np.sum(Y==1)
    num_false = np.sum(Y==0)
    
    entropy_num = -((num_true/num_elements)*np.log2(num_true/num_elements) + 
                    (num_false/num_elements)*np.log2(num_false/num_elements))
    
    return entropy_num

def entropy_given_x(X, Y):
    
    num_elements = Y.shape[0]
    
    feature_true = np.sum(X == 1)
    feature_false = np.sum(X == 0)
    
    p_true = feature_true/ num_elements
    p_false = feature_false/ num_elements
    
    ytrue_xtrue = 0
    yfalse_xtrue = 0
    ytrue_xfalse = 0
    yfalse_xfalse = 0
    
    for index in range(num_elements):
        #X and Y are false
        if X[index] == 0 and Y[index] == 0:
            yfalse_xfalse += 1
        #X is false and Y is true
        if X[index] == 0 and Y[index] == 1:
            ytrue_xfalse += 1
        #X is true and Y is false
        if X[index] == 1 and Y[index] == 0:
            yfalse_xtrue += 1
        #X and Y are true
        if X[index] == 1 and Y[index] == 1:
            ytrue_xtrue += 1
    
    ytrue_xtrue /= feature_true
    yfalse_xtrue /= feature_true
    
    ytrue_xfalse /= feature_false
    yfalse_xfalse /= feature_false
    
    print(yfalse_xtrue)
    print(ytrue_xfalse)
            
    return -(p_true*(ytrue_xtrue*np.log2(ytrue_xtrue)+yfalse_xtrue*np.log2(yfalse_xtrue)) +
             p_false*(ytrue_xfalse*np.log2(ytrue_xfalse)+yfalse_xfalse*np.log2(yfalse_xfalse)))

def information_gain(X, Y):
    
    num_elements = Y.shape[0]
    
    e_y = entropy(Y)
    
    feature_ig = np.zeros(num_elements)
    
    for index in X.shape[0]:
        
        feature = X[:,index]
        feature_ig[index] = entropy_given_x(feature, Y)
        
    return -feature_ig + e_y