
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

    self.bestParam = None 
    self.accur = 0.0 

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
   
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
   
    if self.initialWeightBound is None:
      initBound = 1.0
    else:
      initBound = self.initialWeightBound
    self.W = self.numpRng.uniform(-initBound, initBound, (featureCount, labelCount))
    self.b = self.numpRng.uniform(-initBound, initBound, (labelCount, ))

  def calculateCostAndGradient(self, trainingData, trainingLabels):
    

    cost = 0.0
    big_mat = np.dot(trainingData, self.W) + self.b
    
    
    for label in self.legalLabels:
        indices = np.where(trainingLabels == label)[0]
        for ind in indices:
            cost += big_mat[:,label][ind]
            
    a_max = np.amax(big_mat, axis=1)
    exp_mat = np.exp(np.transpose(np.transpose(big_mat) - a_max))

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
    
    self.W = self.W - learningRate * (grad + l2Reg * self.W)
    self.b = self.b - learningRate * self.b_s
    
        
        

  def validateWeight(self, validationData, validationLabels):
    
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

    guesses = []
    self.posteriors = [] 
    for datum in testData:
      logposterior = self.calculateConditionalProbability(datum)
      guesses.append(np.argmax(logposterior))
      self.posteriors.append(logposterior)

    return guesses
    
  def calculateConditionalProbability(self, datum):
   
    
    bestW, bestb = self.bestParam # These are parameters used for calculating conditional probabilities

    datum_mat = np.matrix(datum)
    prob = []
    
    myus = np.array((datum_mat * np.matrix(bestW))) + bestb 
    myus = myus - np.max(myus)
    myus = np.exp(myus)
    
    prob = myus/myus.sum()
    
    return prob
