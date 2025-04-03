import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, neighbors=5, dist_metric='euclidean'):

        # init classifier with k and distance metric.
        
        self.neighbors = neighbors
        self.dist_metric = dist_metric

    def train(self, data_X, data_y):

        #store training data.
        
        self.data_X = data_X
        self.data_y = data_y

    def _dist(self, pt1, pt2):

        # calc distance between points.
        
        if self.dist_metric == 'euclidean':
 
            return np.sqrt(np.sum((pt1 - pt2) ** 2))
        
        elif self.dist_metric == 'manhattan':
       
            return np.sum(np.abs(pt1 - pt2))
        
        else:
            raise ValueError("Invalid metric")

    def classify(self, test_X):

        # predict labels for test set.
        
        return np.array([self._classify_one(sample) for sample in test_X])

    def _classify_one(self, sample):

        # predict label for single point.
        
        dists = [self._dist(sample, pt) for pt in self.data_X]
        idx = np.argsort(dists)[:self.neighbors]
        labels = [self.data_y[i] for i in idx]
        return Counter(labels).most_common(1)[0][0]

    def accuracy(self, test_X, test_y):

        # calc accuracy of predictions.
        
        preds = self.classify(test_X)
        return np.mean(preds == test_y)
