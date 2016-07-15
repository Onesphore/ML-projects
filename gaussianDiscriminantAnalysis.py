
import math
import sys
import classificationMethod
import numpy as np
import util

class GaussianDiscriminantAnalysisClassifier(classificationMethod.ClassificationMethod):
  def __init__(self, legalLabels, type):
    self.legalLabels = legalLabels
    self.type = type

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels):
    
    
    self.trainingDataSize = trainingData.shape[0]
    self.datumLen = len(trainingData[0])
    

    self.labels_priors = {}
    self.labelsCount = {}
    self.training_labels, priors = np.unique(trainingLabels, return_counts = True)
    
    i = 0
    for label in self.training_labels:
        self.labelsCount.update({label:priors[i]})
        i += 1
    
    priors = np.true_divide(priors, self.trainingDataSize)
    i = 0
    for label in self.training_labels:
        self.labels_priors.update({label:priors[i]})
        i += 1
    

    self.averages = {}
    i = 0
    for label in trainingLabels:
        if label not in self.averages.keys():
            self.averages.update({label:np.array([0.0]*self.datumLen)})
            self.averages[label] = np.add(self.averages[label], trainingData[i])
        else:
            self.averages[label] = np.add(self.averages[label], trainingData[i])
        i += 1
        
    for label in self.averages.keys():
        self.averages[label] = np.true_divide(self.averages[label], self.labelsCount[label])

    self.covariances = {}
    i = 0
    for label in trainingLabels:
        diff = np.subtract(trainingData[i], self.averages[label])
        diff = np.asmatrix(diff)
        val = np.dot(np.transpose(diff), diff)
        if label not in self.covariances.keys():
            self.covariances.update({label:val})
            
            
        else:
            self.covariances[label] = np.add(self.covariances[label], val)
        i += 1
        
    for label in self.covariances.keys():
        self.covariances[label] = np.true_divide(self.covariances[label], self.labelsCount[label])

    
    # LDA shared covariance
    self.sharedCovariance = np.zeros((self.datumLen, self.datumLen))
    for label in self.covariances.keys():
        cov = self.covariances[label].copy()
        self.sharedCovariance += (cov * self.labelsCount[label])
    sh_cov = self.sharedCovariance.copy()
    self.sharedCovariance = np.true_divide(sh_cov, self.trainingDataSize)
    
   
    
    
    # compare LDA's and QDA's accuracy:
    self.betterMethod = "QDA"
    
    #--- QDA ----
    QDA_accuracy = 0
    guesses = self.classify(validationData)
    for guess, label in zip(guesses, validationLabels):
        if guess == label:
            QDA_accuracy += 1
    QDA_accuracy /= float(len(validationLabels))
    
    # --- LDA ---
    self.betterMethod = "LDA"
    
    LDA_accuracy = 0
    guesses = self.classify(validationData)
    for guess, label in zip(guesses, validationLabels):
        if guess == label:
            LDA_accuracy += 1
    LDA_accuracy /= float(len(validationLabels))
    
    if LDA_accuracy < QDA_accuracy:
        self.betterMethod = "QDA"
    
    
    
    

    
  def QDA(self, pi_c, myu_c, sigma_c, x):
    f_sum = np.log(pi_c)
    f_sum += -0.5 * np.log(abs(np.linalg.det(sigma_c)))
    x_m = np.asmatrix(np.subtract(x, myu_c))
    x_m_t = np.transpose(x_m)
    sigmaI = np.linalg.inv(sigma_c)
    f_sum += -0.5 * x_m*sigmaI*x_m_t
    return f_sum

  def LDA(self, pi_c, myu_c, sigma, x):
    f_sum = np.log(pi_c)
    x_mat = np.asmatrix(x)
    x_t = np.transpose(x_mat)
    myuc_mat = np.asmatrix(myu_c)
    myuc_t = np.transpose(myuc_mat)
    sigmaI = np.linalg.inv(sigma)
    f_sum += np.dot(np.dot(x_mat, sigmaI), myuc_t)
    f_sum += -0.5 * myuc_mat*sigmaI*myuc_t
    return f_sum

  def classify(self, testData):

    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      logposterior = self.calculateLogJointProbabilities(datum)
      guesses.append(np.argmax(logposterior))
      self.posteriors.append(logposterior)

    return guesses
    
  def calculateLogJointProbabilities(self, datum):
   
    logJoint = [0 for c in self.legalLabels]
    
    datumMat = np.asmatrix(datum)
    i = 0
    if self.betterMethod == "QDA":
        for label in self.legalLabels:
            pi_c = self.labels_priors[label]
            myu_c = self.averages[label]
            sigma_c = self.covariances[label]
            logJoint[i] = self.QDA(pi_c, myu_c, sigma_c, datumMat)
            i += 1
    else:
        for label in self.legalLabels:
            pi_c = self.labels_priors[label]
            myu_c = self.averages[label]
            sigma = self.sharedCovariance
            logJoint[i] = self.LDA(pi_c, myu_c, sigma, datumMat)
            i += 1
    
    return logJoint
