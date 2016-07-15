
import sys
import classificationMethod
import numpy as np
import util
import scipy.optimize

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SupportVectorMachine(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels, type, data):
        self.legalLabels = legalLabels
        self.type = type

        self.kernel = lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

        self.supportMultipliers = {}
        self.supportVectors = {}
        self.supportVectorLabels = {}
        self.biases = {}
        self.data = data

        if self.data == 'faces':
            self.sigma = 4 
            self.C = 10 
        else:
            self.sigma = 3 
            self.C = 1 


    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        
        for c1 in range(len(self.legalLabels)):
            for c2 in range(c1+1, len(self.legalLabels)):
                # c1: +1 / c2: -1
                X1 = np.asarray([x for i,x in enumerate(trainingData) if trainingLabels[i] == c1]) # Data with label c1
                X2 = np.asarray([x for i,x in enumerate(trainingData) if trainingLabels[i] == c2]) # Data with label c1
                X = np.vstack((X1, X2))
                t = np.concatenate((np.ones(X1.shape[0]), -np.ones(X2.shape[0])))

                lagrangeMultipliers = self.trainSVM(X, t, self.C, self.kernel)
                supportVectorIndices = lagrangeMultipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
                self.supportMultipliers[(c1,c2)] = lagrangeMultipliers[supportVectorIndices]
                self.supportVectors[(c1,c2)] = X[supportVectorIndices]
                self.supportVectorLabels[(c1,c2)] = t[supportVectorIndices]
                self.biases[(c1,c2)] = np.mean(
                    [y - self.predictSVM(
                        x, self.supportMultipliers[(c1,c2)], self.supportVectors[(c1,c2)], self.supportVectorLabels[(c1,c2)], 0, self.kernel
                    )
                    for (x, y) in zip(self.supportVectors[(c1,c2)], self.supportVectorLabels[(c1,c2)])]
                )

    def quadraticProgrammingSolver(self, P, q, G, h, A, b):
        
        func = lambda x, sign=1.0, P=P, q=q: 0.5 * np.dot(np.dot(x, P), x) + np.dot(q, x)
        func_deriv = lambda x, sign=1.0, P=P, q=q: 0.5 * np.dot(P + P.T, x) + q
        constraints = []
        for i in range(A.shape[0]):
            constraints.append({
                'type': 'eq',
                'fun': lambda x, A=A, b=b, i=i: b[i] - np.dot(A[i, :], x),
                'jac': lambda x, A=A, i=i: -A[i, :]
            })
        for i in range(G.shape[0]):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, G=G, h=h, i=i: h[i] - np.dot(G[i, :], x),
                'jac': lambda x, A=A, i=i: -G[i, :]
            })
        cons = tuple(constraints)
        x0 = np.zeros(P.shape[0])
        solution = scipy.optimize.minimize(func, x0, jac=func_deriv, constraints=cons, method='SLSQP')
        return solution.x

    def trainSVM(self, X, t, C, kernel):
        
        N, D = X.shape
        
        
        P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                k = kernel(X[i], X[j])
                P[i][j] = k*t[i] * t[j]
       

        q = -1 * np.ones(N)
        
        G = np.diag(-1 * np.ones(N))
        G = np.vstack((G, np.diag(np.ones(N))))
        
        h = np.zeros(N)
        h = np.append(h, (C * np.ones(N)))
        
        A = np.array([t])
        
        b = np.zeros(1)
      
        a = self.quadraticProgrammingSolver(P, q, G, h, A, b)
       
        
        return a

    def classify(self, testData):
       
        guesses = []
        self.counts = [] 
        for datum in testData:
            count = np.zeros(len(self.legalLabels))
            for c1 in range(len(self.legalLabels)):
                for c2 in range(c1 + 1, len(self.legalLabels)):
                    predict = self.predictSVM(datum,
                                              self.supportMultipliers[(c1,c2)],
                                              self.supportVectors[(c1,c2)],
                                              self.supportVectorLabels[(c1,c2)],
                                              self.biases[(c1,c2)],
                                              self.kernel)
                    if predict > 0:
                        count[c1] += 1
                    else:
                        count[c2] += 1

            guesses.append(np.argmax(count))
            self.counts.append(count)

        return guesses
        
    def predictSVM(self, x, supportMultipliers, supportVectors, supportVectorLabels, bias, kernel):
     
        M, D = supportVectors.shape

        y = 0.0
        for i in range(M):
            k = kernel(supportVectors[i], x)
            y += supportMultipliers[i] * supportVectorLabels[i] * k
        y += np.nan_to_num(bias)

        if y < 0:
            return -1
        else:
            return 1
            

