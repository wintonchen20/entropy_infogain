# Entropy and Information Gain

Entropy: Entropy is a measure of disorder or uncertainty and the goal of machine learning models and Data Scientists in general is to reduce uncertainty. It is also a measure of purity as well.  

Information Gain: We simply subtract the entropy of Y given X from the entropy of just Y to calculate the reduction of uncertainty about Y given an additional piece of information X about Y. This is called Information Gain. The greater the reduction in this uncertainty, the more information is gained about Y from X.  

About this program:  
This is just a quick little program using numpy arrays to calculate entropy and information gain since my machine learning class required calculating so much of them.  

Input:  
X: N x M array where each column is a feature  
Y: N x 1 array that equals the target values 

Returns:  
information_gain array: an array where each element is the information gain of a feature on a target value.  
