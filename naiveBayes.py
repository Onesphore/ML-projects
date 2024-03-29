
import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 
    self.automaticTuning = False 
    
  def setSmoothing(self, k):
    
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
     
      
    self.features = trainingData[0].keys() 
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50] 
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    

    "*** YOUR CODE HERE ***"
    self.dictLabelFeaturesProb = {} # a dictionary to strore all P(Fi=1/Y=y) for different values of k
    self.labelFeaturesProb = {} # a dictionary to store all P(Fi=1/Y=y) for a single value of k
    self.labelsCount = util.Counter() 
    self.labelsProb = util.Counter() 
    
    # evaluate each label's counts and probability
    for label in trainingLabels:
        self.labelsCount[label] += 1
    
    self.trainingDataSize = self.labelsCount.totalCount() # calculate the total number of training data
    self.labelsProb = self.labelsCount.copy()
    self.labelsProb.divideAll(self.trainingDataSize)
   
    
    # for each label calculate features' counts
    for label in self.legalLabels:
        theCounter = util.Counter()
        self.labelFeaturesProb.update({label:theCounter})
    
    for datum, label in zip(trainingData, trainingLabels):
        self.labelFeaturesProb[label].__radd__(datum)
     
    
    #calculate the conditional probabilities
    kPerfor = util.Counter()
    for k in kgrid:
        k_accuracy = 0
        guesses = []
        
        self.featuresProb = {} # this dictionary will store, at the end, all P(Fi=1/Y=y) 
                               # computed with a k value that ensures maximum accuracy  
        
        for label in self.legalLabels:
            theCounter = util.Counter()
            self.featuresProb.update({label:theCounter})
            self.featuresProb[label] = self.labelFeaturesProb[label].copy()
            keyz = self.featuresProb[label].keys()
            self.featuresProb[label].incrementAll(keyz, k)
            self.featuresProb[label].divideAll(self.labelsCount[label] + 2 * k)
        
        
        guesses = self.classify(validationData)
        for g, r in zip(guesses, validationLabels):
            if g == r:
                k_accuracy += 1
        k_accuracy /= float(len(validationLabels))
        kPerfor[str(k)] = k_accuracy
        self.dictLabelFeaturesProb.update({str(k):self.featuresProb})
    
    # makes sure self.featuresProb has P(Fi=1/Y=y), computed with k that ensures maximum accuracy
    maxAcurracyK = kPerfor.argMax()
    for label in self.dictLabelFeaturesProb[maxAcurracyK].keys():
        self.featuresProb[label] = self.dictLabelFeaturesProb[maxAcurracyK][label].copy()
        
   
            
         
             
        
  def classify(self, testData):
    
    guesses = []
    self.posteriors = [] 
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    
    logJoint = util.Counter()
    for label in self.legalLabels:
        logJoint[label]
    for label in self.featuresProb.keys():
        p_y = self.labelsCount[label]/float(self.trainingDataSize)
        logP_y = math.log(p_y)
        sumLogProbFeat = 0
        for feat in self.featuresProb[label].keys():
            if datum[feat] == 1:
                sumLogProbFeat += math.log(self.featuresProb[label][feat])
            else:
                sumLogProbFeat += math.log(1-self.featuresProb[label][feat]) # this probability was just 
                                                                             # inferred 
        logJoint[label] = logP_y + sumLogProbFeat
        
    
    return logJoint
  