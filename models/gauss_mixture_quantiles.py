import numpy as np
import pandas as pd


x_cols = ['TrackP', 'TrackEta', 'NumLongTracks']
y_cols = ['RichDLLbt', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLe']

class Model:
    def fit(self, X, Y, n_bins):
        self.means = {}
        self.stds = {}
        for col in Y.columns:
            self.means[col] = np.zeros((n_bins, n_bins, n_bins))
            self.stds[col] = np.zeros((n_bins, n_bins, n_bins))
        self.bins = {}
        self.masks = {}
        for col in X.columns:
            self.bins[col] = np.quantile(X[col], np.linspace(1./n_bins, 1. - 1./n_bins, n_bins-1))
            left_masks = X[col][:, None] <= self.bins[col]
            right_masks = X[col][:, None] > self.bins[col]
            self.masks[col] = np.zeros((X[col].size, n_bins), dtype=bool)
            self.masks[col][:, 0] = left_masks[:, 0]
            for i in range(n_bins - 2):
                self.masks[col][:, i + 1] = (left_masks[:, i + 1] & right_masks[:, i])
            self.masks[col][:, -1] = right_masks[:, -1]
        
        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    mask = (self.masks['TrackP'][:,i] &
                        self.masks['TrackEta'][:,j] &
                        self.masks['NumLongTracks'][:,k])
                    for col in Y.columns:
                        self.means[col][i,j,k] = np.mean(Y[col][mask])
                        self.stds [col][i,j,k] = np.std (Y[col][mask])
        

    def predict(self, X):
        prediction = pd.DataFrame()
        count = np.zeros((self.means['RichDLLk'].shape), dtype=int)
        pred_masks = {}
        n_bins = count.shape[0]
        
        for col in X.columns:
            left_masks = X[col][:, None] <= self.bins[col]
            right_masks = X[col][:, None] > self.bins[col]
            pred_masks[col] = np.zeros((X[col].size, n_bins), dtype=bool)
            pred_masks[col][:, 0] = left_masks[:, 0]
            for i in range(n_bins - 2):
                pred_masks[col][:, i + 1] = (left_masks[:, i + 1] & right_masks[:, i])
            pred_masks[col][:, -1] = right_masks[:, -1]
        
        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    count[i,j,k] = np.count_nonzero(
                        pred_masks['TrackP'][:,i] &
                        pred_masks['TrackEta'][:,j] &
                        pred_masks['NumLongTracks'][:,k])
        
        
        for col in self.means.keys():
            gaussian = np.array([])
            for i in range(n_bins):
                for j in range(n_bins):
                    for k in range(n_bins):
                        local = np.random.normal(loc=self.means[col][i,j,k],
                                                 scale=self.stds[col][i,j,k],
                                                 size=count[i,j,k])
                        gaussian = np.append(gaussian, local)
                        
            prediction[col] = gaussian
        return prediction

