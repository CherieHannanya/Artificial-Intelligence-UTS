import numpy as np
from collections import Counter

class node:
        def __init__(self, feature =None, threshold=None, letf=None, right=None,*, value=None):
            self.feature = feature
            self.threshold =threshold
            self.left = letf      
            self.right = right
            self.value=value

        def is_leaf_node(self):
            return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y,):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y,  depth=0): 
        n_sampel, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth>=self.max_depth or n_labels==1 or n_sampel<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return node(value=leaf_value)

        feat_idx = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh  = self._best_split(X, y, feat_idx)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs : 
            X_colum = X[:, feat_idx]
            threshold = np.unique(X_colum)

            for thr in threshold:
                gain = self._information_gain(y, X_colum, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return  split_idx, split_threshold

    def _information_gain(self,y, X_colum, threshold):
        #parent entropy 
        parennt_entropy = self._entropy(y)
        #cildren 
        left_idx, right_idx = self._split(X_colum, threshold)
        if len(left_idx) == 0 or len(right_idx) ==0:
         return 0

        #calculate weighted
        n = len(y)
        n_l, e_r = len(left_idx), len(right_idx)
        e_l, n_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #ig
        information_gain = parennt_entropy - child_entropy
        return information_gain
     #


    def _split(self, X_colum, split_thresh):
        left_idxs = np.argwhere(X_colum <= split_thresh).flatten()
        right_idxs = np.argwhere(X_colum > split_thresh).flatten()
        return left_idxs, right_idxs 

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])


    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
            
