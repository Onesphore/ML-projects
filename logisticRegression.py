# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 2016

@author: jphong
"""
import classificationMethod
import numpy as np
import util
import itertools

class LogisticRegressionClassifier(classificationMethod.ClassificationMethod):
  def __init__(self, legalLabels, type, seed):
    self.legalLabels = legalLabels
    self.type = type
    self.learningRate = [0.01, 0.001, 0.0001]
    self.l2Regularize = [1.0, 0.1, 0.0]
    self.numpRng = np.random.RandomState(seed)
    self.initialWeightBound = None
    self.posteriors = []
    self.costs = []
    self.epoch = 1000

    self.bestParam = None # You must fill in this variable in validateWeight
    self.accur = 0.0 # added by me (Ones)

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method.
    Iterates several learning rates and regularization parameter to select the best parameters.

    Do not modify this method.
    """
    for lRate in self.learningRate:
      curCosts = []
      for l2Reg in self.l2Regularize:
        self.initializeWeight(trainingData.shape[1], len(self.legalLabels))
        for i in xrange(self.epoch):
          cost, grad = self.calculateCostAndGradient(trainingData, trainingLabels)
          self.updateWeight(grad, lRate, l2Reg)
          curCosts.append(cost)
        self.validateWeight(validationData, validationLabels)
        self.costs.append(curCosts)
        

  def initializeWeight(self, featureCount, labelCount):
    """
    Initialize weights and bias with randomness.

    Do not modify this method.
    """
    if self.initialWeightBound is None:
      initBound = 1.0
    else:
      initBound = self.initialWeightBound
    self.W = self.numpRng.uniform(-initBound, initBound, (featureCount, labelCount))
    self.b = self.numpRng.uniform(-initBound, initBound, (labelCount, ))

  def calculateCostAndGradient(self, trainingData, trainingLabels):
    """
    Fill in this function!

    trainingData : (N x D)-sized numpy array
    trainingLabels : N-sized list
    - N : the number of training instances
    - D : the number of features (PCA was used for feature extraction)
    RETURN : (cost, grad) python tuple
    - cost: python float, negative log likelihood of training data
    - grad: gradient which will be used to update weights and bias (in updateWeight)

    Evaluate the negative log likelihood and its gradient based on training data.
    Gradient evaluted here will be used on updateWeight method.
    Note the type of weight matrix and bias vector:
    self.W : (D x C)-sized numpy array
    self.b : C-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    - C : the number of legal labels
    """

    "*** YOUR CODE HERE ***"
    # cost a.k.a NLL
    cost = 0.0
    big_mat = np.dot(trainingData, self.W) + self.b
    
    
    for label in self.legalLabels:
        indices = np.where(trainingLabels == label)[0]
        for ind in indices:
            cost += big_mat[:,label][ind]
            
    a_max = np.amax(big_mat, axis=1)
    exp_mat = np.exp(np.transpose(np.transpose(big_mat) - a_max))
    #exp_mat = np.exp(big_mat - np.transpose(a_max))
    sum_exp = np.log(np.sum(exp_mat, axis = 1))
    
    cost -= np.sum(a_max)
    cost -= np.sum(sum_exp)
    cost = -cost
    
    # gradient
    self.b_s = np.zeros(len(self.W[0]))
    grad = np.zeros((len(self.W), len(self.legalLabels)))
    myuc = exp_mat/exp_mat.sum(axis=1)[:,None]
    yic = np.zeros((len(trainingLabels), len(self.legalLabels)))
    for i in range (len(trainingLabels)):
        np.put(yic[i,:], trainingLabels[i], 1)
        x_mat = np.matrix(trainingData[i,:])
        diff = np.subtract(myuc[i,:], yic[i,:])
        diff_mat = np.matrix(diff)
        grad += np.array(np.transpose(x_mat) * diff_mat)
        self.b_s += diff
                      
    return cost, grad

  def updateWeight(self, grad, learningRate, l2Reg):
    """
    Fill in this function!
    grad : gradient which was evaluated in calculateCostAndGradient
    learningRate : python float, learning rate for gradient descent
    l2Reg: python float, L2 regularization parameter

    Update the logistic regression parameters using gradient descent.
    Update must include L2 regularization.
    Please note that bias parameter must not be regularized.
    """

    "*** YOUR CODE HERE ***"
    #w_copy = np.copy(self.W)
    #b_copy = np.copy(self.b)
    
    self.W = self.W - learningRate * (grad + l2Reg * self.W)
    self.b = self.b - learningRate * self.b_s
    
        
        

  def validateWeight(self, validationData, validationLabels):
    """
    Fill in this function!

    validationData : (M x D)-sized numpy array
    validationLabels : M-sized list
    - M : the number of validation instances
    - D : the number of features (PCA was used for feature extraction)

    Choose the best parameters of logistic regression.
    Calculates the accuracy of the validation set to select the best parameters.
    """

    "*** YOUR CODE HERE ***"
    
    if self.bestParam != None:
        w, b = self.bestParam
        w_old = np.copy(w)
        b_old = np.copy(b)
    elif self.bestParam == None:
        pass
    
    self.bestParam = (self.W, self.b)
        
    count = 0
    guesses = self.classify(validationData)
    for g, label in zip(guesses, validationLabels):
        if g == label:
            count += 1
    
    acc = count/float(len(validationLabels))
    
    if acc >= self.accur:
        self.accur = acc
    else:
        self.bestParam = (w_old, b_old)


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    Do not modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      logposterior = self.calculateConditionalProbability(datum)
      guesses.append(np.argmax(logposterior))
      self.posteriors.append(logposterior)

    return guesses
    
  def calculateConditionalProbability(self, datum):
    """
    datum : D-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    RETURN : C-sized numpy array
    - C : the number of legal labels

    Returns the conditional probability p(y|x) to predict labels for the datum.
    Return value is NOT the log of probability, which means 
    sum of your calculation should be 1. (sum_y p(y|x) = 1)
    """
    
    bestW, bestb = self.bestParam # These are parameters used for calculating conditional probabilities

    "*** YOUR CODE HERE ***"
    datum_mat = np.matrix(datum)
    prob = []
    
    myus = np.array((datum_mat * np.matrix(bestW))) + bestb 
    myus = myus - np.max(myus)
    myus = np.exp(myus)
    
    prob = myus/myus.sum()
    
    return prob
