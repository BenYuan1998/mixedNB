#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:34:27 2022

@author: nuoyuan
"""


# %% Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import norm # norm.pdf(x, loc, scale) computes the probability density of x realized from a univariate Gaussian distribution with mean = loc, std = scale
from scipy.stats import multivariate_normal # multivariate_normal(x, mean, cov) computes the probability density of x realizsed from a multivariate Gaussian distribution with mean vector = mean and covariance matrix = cov
from numpy.linalg import inv # compute the inverse of a nonsingular square matrix
from numpy.linalg import pinv # compute the pseudo inverse in case the empirical scatter matrix is not invertible
from sklearn.metrics import accuracy_score  

# %% Necessary data preprocessing and partitioning (Remember to change the current working directory to the directory of the folder hosting this script)
path = "./dataset_31_credit-g.csv" # the dataset of interest should be placed in the same folder as this script
data = pd.read_csv(path, header = 0)
target_labelencoder = {"good" : 1, "bad" : 0}
y = data["class"].map(target_labelencoder) # Label encode the target variable and store it as a separate Series
X = data.drop(["class"], axis = 1) # Store all the attributes as a separate DataFrame
categorical_attributes = X.select_dtypes(include=["object"]).columns # Identify the categorical attributes
continuous_attributes = X.select_dtypes(include=["int"]).columns # Identify the continuous attributes
X[continuous_attributes] = X[continuous_attributes].astype("float") # Convert the values of all the continuous attributes to float
X[categorical_attributes] = X[categorical_attributes].apply(LabelEncoder().fit_transform) # Label encode all the categorical attributes
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3) # Partition the original dataset into a training set and a test set by the 70%-30% ratio

# %% A class for model training and predictive decision making using the mixed Naive Bayes classifier. Two point-estimate methods (i,e., MLE and MAP) are available for model fitting.
class MixedNB():
    """
    Parameters
    ----------
    continuous_attributes : pandas columns
         The labels of the columns corresponding to the continuous attributes. 
    categorical_attributes : pandas columns
         The labels of the columns corresponding to the categorical attributes.
         
    Attributes
    ----------
    uni_gaussian_MLE: dict
        a dictionary used to store the MLE estimates for the mean and variance associated with each class-conditional univariate Gaussian distribution.
    multi_gaussian_MLE: dict
        a dictionary used to store the MLE estimates for the mean and covariance matrix associated with each class-conditional multivariate Gaussian distribution.
    categorical_MLE: dict
        a dictionary used to store the MLE estimates for the category probabilities associated with each class-conditional categorical distribution.
    class_MLE: 
        a dictionary used to store the MLE estimates for the class probabilities associated with the target attribute.
    uni_gaussian_MAP : dict
        a dictionary used to store the MAP estimates for the mean and precision associated with each class-conditional univariate Gaussian distribution.
    multi_gaussian_MAP: dict
        a dictionary used to store the MAP estimates for the mean and covariance matrix associated with each class-conditional multivariate Gaussian distribution.
    categorical_MAP : dict
        a dictionary used to store the MAP estimates for the category probabilities associated with each class-conditional categorical distribution.
    class_MAP : dict
        a dictionary used to store the MAP estimates for the class probabilities associated with the target attribute.
    """
    def __init__(self, continuous_attributes, categorical_attributes):
        self.continuous_attributes = continuous_attributes
        self.categorical_attributes = categorical_attributes
        self.uni_gaussian_MLE = {1:{}, 0:{}}
        self.multi_gaussian_MLE = {1:{}, 0:{}}
        self.categorical_MLE = {1:{}, 0:{}}
        self.class_MLE = {}
        self.uni_gaussian_MAP = {1:{}, 0:{}}
        self.multi_gaussian_MAP = {1:{}, 0:{}}
        self.categorical_MAP = {1:{}, 0:{}}
        self.class_MAP = {}
   
    def separateByClass(self, X_train, y_train):
        """
        Store training samples belong to each class to a separate DataFrame.
        
        Parameters
        ----------
        X_train : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains the feature values of all the training samples
        y_train : pandas Series, shape = [num_samples,]
            a pandas Series that contains the target value of all the training samples
            
        Returns
        -------
        X_positive : pandas DataFrame
            a pandas DataFrame that contains all the positive training samples
        num_positive: int
            number of positive samples
        X_negative : pandas DataFrame
            a pandas DataFrame that contains all the negative training samples
        num_negative: int
            number of negative samples
        
        """
        positive_indices = y_train.loc[y_train == 1].index.tolist()
        negative_indices = y_train.loc[y_train == 0].index.tolist()
        X_positive = X_train.loc[positive_indices]
        num_positive = X_positive.shape[0]
        X_negative = X_train.loc[negative_indices]
        num_negative = X_negative.shape[0]
        return X_positive, num_positive, X_negative, num_negative
    
    def UniGaussianMLE(self, positive_samples, num_positive_samples, negative_samples, num_negative_samples):
        """
        Compute the MLE estimates for the mean and precision associated with each class-conditional univariate Gaussian distribution.
        This function is invoked during model training when boolean argument naive_assumption in function fit is set to be True.
        
        Parameters
        ----------
        positive_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the positive training samples
        num_positive_samples: int
            number of positive samples
        negative_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the negative training samples
        num_negative_samples: int
            number of negative samples
        Returns
        -------
        None.
        """
        X_positive_continuous = positive_samples[self.continuous_attributes]
        X_negative_continuous = negative_samples[self.continuous_attributes]
        for attribute in self.continuous_attributes:
            sum_positive = np.sum(X_positive_continuous[attribute])
            mu_MLE_positive = sum_positive / num_positive_samples
            variance_MLE_positive = np.sum(X_positive_continuous[attribute].map(lambda x: np.square(x - mu_MLE_positive))) / num_positive_samples
            sum_negative = np.sum(X_negative_continuous[attribute])
            mu_MLE_negative = sum_negative / num_negative_samples
            variance_MLE_negative = np.sum(X_negative_continuous[attribute].map(lambda x: np.square(x - mu_MLE_negative))) / num_negative_samples
            self.uni_gaussian_MLE[1][attribute] = {"mu": mu_MLE_positive, "variance": variance_MLE_positive}
            self.uni_gaussian_MLE[0][attribute] = {"mu": mu_MLE_negative, "variance": variance_MLE_negative}
            
    def MultiGaussianMLE(self, positive_samples, num_positive_samples, negative_samples, num_negative_samples):
        """
        Compute the MLE estimates for the mean and covariance matrix associated with each class-conditional multivariate Gaussian distribution.
        This function is invoked during model training when boolean argument naive_assumption in function fit is set to be True.
        
        Parameters
        ----------
        positive_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the positive training samples
        num_positive_samples: int
            number of positive samples
        negative_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the negative training samples
        num_negative_samples: int
            number of negative samples
        -------
        None.

        """
        X_positive_continuous = positive_samples[self.continuous_attributes]
        X_negative_continuous = negative_samples[self.continuous_attributes]
        X_positive_continuous = X_positive_continuous.to_numpy() # Convert to a multidimensional numpy array convenient for elementwise operations
        X_negative_continuous = X_negative_continuous.to_numpy() # Convert to a multidimensional numpy array convenient for elementwise operations
        mean_positive_MLE = np.sum(X_positive_continuous, axis = 0, keepdims = True) / num_positive_samples
        mean_negative_MLE = np.sum(X_negative_continuous, axis = 0, keepdims = True) / num_positive_samples
        cov_positive_MLE = 0 # initialization
        cov_negative_MLE = 0 # initializaiton
        for record in X_positive_continuous:
            cov_positive_MLE += np.dot((record - mean_positive_MLE).T,  (record - mean_positive_MLE))
        for record in X_negative_continuous:
            cov_negative_MLE += np.dot((record - mean_negative_MLE).T,  (record - mean_negative_MLE))
        cov_positive_MLE = cov_positive_MLE / num_positive_samples
        cov_negative_MLE = cov_negative_MLE / num_negative_samples
        self.multi_gaussian_MLE[1]["mean"] = mean_positive_MLE
        self.multi_gaussian_MLE[1]["cov"] = cov_positive_MLE
        self.multi_gaussian_MLE[0]["mean"] = mean_negative_MLE
        self.multi_gaussian_MLE[0]["cov"] = cov_negative_MLE
        
    def CategoricalMLE(self, positive_samples, num_positive_samples, negative_samples, num_negative_samples):
        
        """
         Compute the MLE estimates for the category probabilities associated with each class-conditional categorical distribution.
         
        Parameters
        ----------
        positive_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the positive training samples
        num_positive_samples: int
            number of positive samples
        negative_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the negative training samples
        num_negative_samples: int
            number of negative samples

        Returns
        -------
        None.

        """
        X_positive_categorical = positive_samples[self.categorical_attributes]
        X_negative_categorical = negative_samples[self.categorical_attributes]
        X_categorical = pd.concat([X_positive_categorical, X_negative_categorical], axis = 0)
        for attribute in self.categorical_attributes:
            self.categorical_MLE[1][attribute] = {}
            self.categorical_MLE[0][attribute] = {}
            categories = np.unique(X_categorical[attribute])
            category_freq_positive = {}
            category_freq_negative = {}
            for category in categories:
                category_freq_positive[category] = 0
                category_freq_negative[category] = 0
            for value in X_positive_categorical[attribute].tolist():
                category_freq_positive[value] += 1
            for value in X_negative_categorical[attribute].tolist():
                category_freq_negative[value] += 1
            for category, freq in category_freq_positive.items():
                self.categorical_MLE[1][attribute][category] = freq / num_positive_samples
            for category, freq in category_freq_negative.items():
                self.categorical_MLE[0][attribute][category] = freq / num_negative_samples
       
    def ClassMLE(self, y):
        """
        Compute the MLE estimates for the class probabilities of the target attribute.
        
        Parameters
        ----------
        y : pandas Series, shape = [num_samples,]
        
        Returns
        -------
        None.

        """
        num_samples = y.shape[0]
        positive_MLE = y.loc[y == 1].shape[0] / num_samples
        negative_MLE = y.loc[y == 0].shape[0] / num_samples
        self.class_MLE[1] = positive_MLE
        self.class_MLE[0] = negative_MLE
    
    
    def UniGaussianMAP(self, positive_samples, num_positive_samples, negative_samples, num_negative_samples, parameters):
        """
        Compute the MAP estimates for the mean and precision associated with each class-conditional univariate Gaussian distribution.
        This function is invoked during model training when boolean argument naive_assumption in function fit is set to be True.
        
        Parameters
        ----------
        positive_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the positive training samples
        num_positive_samples: int
            number of positive samples
        negative_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the negative training samples
        num_negative_samples: int
            number of negative samples
         parameters : dict()
            a dictionary of mu_0, lambda_0, alpha_0, beta_0 and their pre-specified values, the four parameters parameterizing a normal-gamma distribution. The default value is {"mu_0": 0, "lambda_0" : 2, "alpha_0" : 0, "beta_0": 0}
        Returns
        -------
        None.
        """
        mu_0 = parameters["mu_0"]
        lambda_0 = parameters["lambda_0"]
        alpha_0 = parameters["alpha_0"]
        beta_0 = parameters["beta_0"]
        X_positive_continuous = positive_samples[self.continuous_attributes]
        X_negative_continuous = negative_samples[self.continuous_attributes]
        for attribute in self.continuous_attributes:
            sum_positive = np.sum(X_positive_continuous[attribute])
            mu_MAP_positive = (sum_positive + lambda_0 * mu_0) / (num_positive_samples + lambda_0)
            deviation_positive = np.sum(X_positive_continuous[attribute].map(lambda x: np.square(x - mu_MAP_positive)))
            precision_MAP_positive =  (1 + 2 * alpha_0 + num_positive_samples - 2) / (2 * beta_0 + lambda_0 * np.square(mu_MAP_positive - mu_0) + deviation_positive)
            sum_negative = np.sum(X_negative_continuous[attribute])
            mu_MAP_negative = (sum_negative + lambda_0 * mu_0) / (num_negative_samples + lambda_0)
            deviation_negative = np.sum(X_negative_continuous[attribute].map(lambda x: np.square(x - mu_MAP_negative)))
            precision_MAP_negative =  (1 + 2 * alpha_0 + num_negative_samples - 2) / (2 * beta_0 + lambda_0 * np.square(mu_MAP_negative - mu_0) + deviation_negative)
            self.uni_gaussian_MAP[1][attribute] = {"mu": mu_MAP_positive, "precision": precision_MAP_positive}
            self.uni_gaussian_MAP[0][attribute] = {"mu": mu_MAP_negative, "precision": precision_MAP_negative}
            
    def MultiGaussianMAP(self, positive_samples, num_positive_samples, negative_samples, num_negative_samples, parameters):
        """
        Compute the MAP estimates for the mean and covariance matrix associated with each class-conditional multivariate Gaussian distribution.
        This function is invoked during model training when boolean argument naive_assumption in function fit is set to be True.
        
        Parameters
        ----------
        positive_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the positive training samples
        num_positive_samples: int
            number of positive samples
        negative_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the negative training samples
        num_negative_samples: int
            number of negative samples
        parameters: dict()
             a dictionary of mu_0, k_0 and v_0, the mean of the Normal prior and two strength parameters parameterizing a Normal-inverse-Wishart distribution and their pre-specified values. The default value is {"mu_0" : 0, "k_0" : 5, "v_0" : 5}
        alpha_categorical_attribute: float
        -------
        None.

        """
        mu_0 = parameters["mu_0"]
        k_0 = parameters["k_0"]
        v_0 = parameters["v_0"]
        num_continuous_attributes = len(self.continuous_attributes)
        X_positive_continuous = positive_samples[self.continuous_attributes]
        X_negative_continuous = negative_samples[self.continuous_attributes]
        X_positive_continuous = X_positive_continuous.to_numpy() # Convert to a multidimensional numpy array convenient for elementwise operations
        X_negative_continuous = X_negative_continuous.to_numpy() # Convert to a multidimensional numpy array convenient for elementwise operations
        mean_positive_MAP = np.sum(X_positive_continuous, axis = 0, keepdims = True) + k_0 * mu_0 * num_positive_samples + k_0
        mean_negative_MAP = np.sum(X_negative_continuous, axis = 0, keepdims = True) + k_0 * mu_0 * num_negative_samples + k_0
        empirical_mean_positive = np.sum(X_positive_continuous, axis = 0, keepdims = True)
        empirical_mean_negative = np.sum(X_negative_continuous, axis = 0, keepdims = True)
        empirical_scatter_matrix_positive = 0 # initialization
        empirical_scatter_matrix_negative = 0 # initialization
        for record in X_positive_continuous:
            empirical_scatter_matrix_positive += np.dot((record - empirical_mean_positive).T, (record - empirical_mean_negative))
        for record in X_negative_continuous:
            empirical_scatter_matrix_negative += np.dot((record - empirical_mean_negative).T, (record - empirical_mean_negative))
        # print(empirical_scatter_matrix_positive.shape, empirical_scatter_matrix_negative.shape)
        MAP_scatter_matrix_positive = 0 # initialization
        MAP_scatter_matrix_negative = 0 # initialization
        for record in X_positive_continuous:
            MAP_scatter_matrix_positive += np.dot((record - mean_positive_MAP).T, (record - mean_positive_MAP))
        for record in X_negative_continuous:
            MAP_scatter_matrix_negative += np.dot((record - mean_negative_MAP).T, (record - mean_negative_MAP))
        #print(MAP_scatter_matrix_positive.shape, MAP_scatter_matrix_negative.shape)
        try: # check whether empirical_scatter_matrix_positive is invertible. If it is, use its inverse to compute cov_positive_MAP; otherwise, use its pseudo inverse
            cov_positive_MAP = (MAP_scatter_matrix_positive + k_0 * np.dot((mean_positive_MAP - mu_0).T, (mean_positive_MAP - mu_0)) + inv(empirical_scatter_matrix_positive)) / (num_positive_samples + v_0 - num_continuous_attributes)
        except:
            cov_positive_MAP = (MAP_scatter_matrix_positive + k_0 * np.dot((mean_positive_MAP - mu_0).T, (mean_positive_MAP - mu_0)) + pinv(empirical_scatter_matrix_positive)) / (num_positive_samples + v_0 - num_continuous_attributes)
        try: # in the same vein as the previous try/except
            cov_negative_MAP = (MAP_scatter_matrix_negative + k_0 * np.dot((mean_negative_MAP - mu_0).T, (mean_negative_MAP - mu_0)) + inv(empirical_scatter_matrix_negative)) / (num_negative_samples + v_0 - num_continuous_attributes)
        except:
            cov_negative_MAP = (MAP_scatter_matrix_negative + k_0 * np.dot((mean_negative_MAP - mu_0).T, (mean_negative_MAP - mu_0)) + pinv(empirical_scatter_matrix_negative)) / (num_negative_samples + v_0 - num_continuous_attributes)
        self.multi_gaussian_MAP[1]["mean"] = mean_positive_MAP
        self.multi_gaussian_MAP[1]["cov"] = cov_positive_MAP
        self.multi_gaussian_MAP[0]["mean"] = mean_negative_MAP
        self.multi_gaussian_MAP[0]["cov"] = cov_negative_MAP
        
    def CategoricalMAP(self, positive_samples, num_positive_samples, negative_samples, num_negative_samples, alpha):
        """
         Compute the MAP estimates for the category probabilities associated with each class-conditional categorical distribution.
         
        Parameters
        ----------
        positive_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the positive training samples
        num_positive_samples: int
            number of positive samples
        negative_samples : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains all the negative training samples
        num_negative_samples: int
            number of negative samples
         alpha : float
            the concentration parameter corresponding to each category of a categorical attribute. The default value is 5. In this case, the mode of the Dirichlet distribution is the uniform distribution (alpha is the same across categories). Moreover, the greater the sum of alpha's, the greater the probability mass placed over the uniform distribution is.

        Returns
        -------
        None.

        """
        X_positive_categorical = positive_samples[self.categorical_attributes]
        X_negative_categorical = negative_samples[self.categorical_attributes]
        X_categorical = pd.concat([X_positive_categorical, X_negative_categorical], axis = 0)
        for attribute in self.categorical_attributes:
            self.categorical_MAP[1][attribute] = {}
            self.categorical_MAP[0][attribute] = {}
            categories = np.unique(X_categorical[attribute])
            num_categories = len(categories)
            category_freq_positive = {}
            category_freq_negative = {}
            for category in categories:
                category_freq_positive[category] = 0
                category_freq_negative[category] = 0
            for value in X_positive_categorical[attribute].tolist():
                category_freq_positive[value] += 1
            for value in X_negative_categorical[attribute].tolist():
                category_freq_negative[value] += 1
            for category, freq in category_freq_positive.items():
                self.categorical_MAP[1][attribute][category] = (freq + alpha - 1) / (num_positive_samples + num_categories * (alpha - 1))
            for category, freq in category_freq_negative.items():
                self.categorical_MAP[0][attribute][category] = (freq + alpha - 1) / (num_negative_samples + num_categories * (alpha - 1))
    
    def ClassMAP(self, y, alpha):
        """
        Compute the MAP estimates for the class probabilities of the target attribute.
        
        Parameters
        ----------
        y : pandas Series, shape = [num_samples,]
        alpha : float
            the concentration parameter corresponding to each category of the target attribute. The default value is 5.
        
        Returns
        -------
        None.

        """
        num_samples = y.shape[0]
        positive_MAP = (y.loc[y == 1].shape[0] + alpha - 1) / (num_samples + 2 * (alpha - 1))
        negative_MAP = (y.loc[y == 0].shape[0] + alpha - 1) / (num_samples + 2 * (alpha - 1))
        self.class_MAP[1] = positive_MAP
        self.class_MAP[0] = negative_MAP
        
    def fit_MLE(self, X, y, naive_assumption = True):
        """
        Fit the mixed Naive Bayes classifier using the MLE method. 
        
        Parameters
        ----------
        X : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains the feature values of all the training samples
        y : pandas Series, shape = [num_samples,]
            a pandas Series that contains the class values of all the training samples
        naive_assumption : boolean
            If True (the default value), each continuous feature is indepdent of one another conditional on class and consequently is modeled as a univariate Gaussian for each class; othwrwise, all the continuous features are jointly modeled by a multivariate Gaussian for each class.
        
        Returns
        -------
        None.
        """
        X_positive, num_positive_samples, X_negative, num_negative_samples = self.separateByClass(X, y)
        if naive_assumption:
            self.UniGaussianMLE(X_positive, num_positive_samples, X_negative, num_negative_samples)
        else:
            self.MultiGaussianMLE(X_positive, num_positive_samples, X_negative, num_negative_samples)
        self.CategoricalMLE(X_positive, num_positive_samples, X_negative, num_negative_samples)
        self.ClassMLE(y)
        
    def fit_MAP(self, X, y, NG_parameters = {"mu_0": 0, "lambda_0" : 2, "alpha_0" : 1, "beta_0": 1}, NIW_parameters = {"mu_0" : 0 ,"k_0" : 5, "v_0" : 5}, alpha_categorical_attribute = 5, alpha_target_attribute = 5, naive_assumption = True):
        """
        Fit the mixed Naive Bayes classifier using the MAP method.
        
        Parameters
        ----------
        X : pandas DataFrame, shape = [num_samples, n_features]
            a pandas DataFrame that contains the feature values of all the training samples
        y : pandas Series, shape = [num_samples,]
            a pandas Series that contains the class values of all the training samples
        NG_parameters : dict()
            a dictionary of mu_0, lambda_0, alpha_0, beta_0 parameterizing a Normal-gamma distribution and their pre-specified values. The default value is {"mu_0": 0, "lambda_0" : 2, "alpha_0" : 1, "beta_0": 1}. Moreover, S_0, the scale matrix parameter, is set to be equal to the empirical scatter matrix.
        NIW_parameters: dict()
            a dictionary of mu_0, k_0 and v_0, the mean of the Normal prior and two strength parameters parameterizing a Normal-inverse-Wishart distribution and their pre-specified values. The default value is {"mu_0" : 0, "k_0" : 5, "v_0" : 5}
        alpha_categorical_attribute: float
        the concentration parameter corresponding to each category of a categorical attribute. The default value is 5.
        alpha_target_attribute: float
        the concentration parameter corresponding to each category of the target attribute. The default value is 5.
        naive_assumption : boolean
            If True (the default value), each continuous feature is indepdent of one another conditional on class and consequently is modeled as a univariate Gaussian for each class; othwrwise, all the continuous features are jointly modeled by a multivariate Gaussian for each class.
        
        Returns
        -------
        None.
        """
        X_positive, num_positive_samples, X_negative, num_negative_samples = self.separateByClass(X, y)
        if naive_assumption:
            self.UniGaussianMAP(X_positive, num_positive_samples, X_negative, num_negative_samples, NG_parameters)
        else:
            self.MultiGaussianMAP(X_positive, num_positive_samples, X_negative, num_negative_samples, NIW_parameters)
        self.CategoricalMAP(X_positive, num_positive_samples, X_negative, num_negative_samples, alpha_categorical_attribute)
        self.ClassMAP(y, alpha_target_attribute)
        
    def log_categorical_mass_MLE(self, categorical_attribute, target_class): 
        """
        A util function that can be used to compute the probability masses associated with an i.i.d. sample drawn from a class-conditional categorical distribution whose parameters are estimated using MLE.
            
        Parameters
        ----------
        categorical_attribute : pandas Series, shape = [num_samples,]
        a pandas Series that stores the sample 
        target_class : int
        the class of interest

        Returns
        -------
        log_masses: pandas Series, shape = [num_samples,]
        a pandas Series that stores the corresponding probability masses
        """
        
        label = categorical_attribute.name
        index = categorical_attribute.index
        log_masses = list()
        for value in categorical_attribute.tolist():
            log_masses.append(np.log(self.categorical_MLE[target_class][label][value]))
        return pd.Series(log_masses, index = index, name = label)
        
    def log_categorical_mass_MAP(self, categorical_attribute, target_class): 
        """
        A util function that can be used to compute the probability masses associated with an i.i.d. sample drawn from a class-conditional categorical distribution whose parameters are estimated using MAP.
            
        Parameters
        ----------
        categorical_attribute : pandas Series, shape = [num_samples,]
        a pandas Series that stores the sample 
        target_class : int
        the class of interest

        Returns
        -------
        log_masses: pandas Series, shape = [num_samples,]
        a pandas Series that stores the corresponding probability masses
        """
        
        label = categorical_attribute.name
        index = categorical_attribute.index
        log_masses = list()
        for value in categorical_attribute.tolist():
            log_masses.append(np.log(self.categorical_MAP[target_class][label][value]))
        return pd.Series(log_masses, index = index, name = label)
    
    def predict_proba_MLE_naive_assumption(self, X_test):
        """
        Returns probability estimates for the test vector X_test assuming the naive assumption among the continuous attributes using a mixed Naive Bayes classifier fit using MLE
        
        Parameters
        ----------
        X_test: pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors
        
        Returns
        -------
        C: pandas DataFrame, shape = [num_samples, num_classes]
            a pandas DataFrame that stores logP(x, y = 0/1| phi_MLE), values proportional to the probabilities for positive and negative
        
        """
        num_samples = X_test.shape[0]
        X_continuous = X_test[self.continuous_attributes]
        X_categorical = X_test[self.categorical_attributes]
        X_log_densities_positive = X_continuous.apply(lambda x: np.log(norm.pdf(x, loc = self.uni_gaussian_MLE[1][x.name]["mu"]  , scale = np.sqrt(self.uni_gaussian_MLE[1][x.name]["variance"]))))
        X_log_densities_negative = X_continuous.apply(lambda x: np.log(norm.pdf(x, loc = self.uni_gaussian_MLE[0][x.name]["mu"]  , scale = np.sqrt(self.uni_gaussian_MLE[0][x.name]["variance"]))))
        X_log_masses_positive = X_categorical.apply(self.log_categorical_mass_MLE, args = (1,))
        X_log_masses_negative = X_categorical.apply(self.log_categorical_mass_MLE, args = (0,))
        log_mass_class_positive = pd.DataFrame({"1" : [np.log(self.class_MLE[1])] * num_samples})
        log_mass_class_negative = pd.DataFrame({"0" : [np.log(self.class_MLE[0])] * num_samples})
        X_log_densities_positive.reset_index(drop = True, inplace = True)
        X_log_densities_negative.reset_index(drop = True, inplace = True)
        X_log_masses_positive.reset_index(drop = True, inplace = True)
        X_log_masses_negative.reset_index(drop = True, inplace = True)
        log_mass_class_positive.reset_index(drop = True, inplace = True)
        log_mass_class_negative.reset_index(drop = True, inplace = True)
        X_concat_positive = pd.concat([X_log_densities_positive, X_log_masses_positive, log_mass_class_positive], axis = 1)
        X_concat_negative = pd.concat([X_log_densities_negative, X_log_masses_negative, log_mass_class_negative], axis = 1)
        X_sum_positive = X_concat_positive.sum(axis = 1).tolist()
        X_sum_negative = X_concat_negative.sum(axis = 1).tolist()
        C = pd.DataFrame({1: X_sum_positive, 0 : X_sum_negative})
        return C
    
    def predict_proba_MLE(self, X_test):
        """
        Returns probability estimates for the test vector X_test without assuming the naive assumption among the continuous attributes using a mixed Naive Bayes classifier fit using MLE
        
        Parameters
        ----------
        X_test: pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors
        
        Returns
        -------
        C: pandas DataFrame, shape = [num_samples, num_classes]
            a pandas DataFrame that stores logP(x, y = 0/1| phi_MLE), values proportional to the probabilities for positive and negative
        
        """
        num_samples = X_test.shape[0]
        X_continuous = X_test[self.continuous_attributes]
        X_continuous = X_continuous.to_numpy()
        X_categorical = X_test[self.categorical_attributes]
        X_log_densities_positive = list()
        X_log_densities_negative = list()
        for record in X_continuous:
            X_log_densities_positive.append(np.log(multivariate_normal.pdf(record.ravel(), self.multi_gaussian_MLE[1]["mean"].ravel(), self.multi_gaussian_MLE[1]["cov"], allow_singular=True)))
            X_log_densities_negative.append(np.log(multivariate_normal.pdf(record.ravel(), self.multi_gaussian_MLE[0]["mean"].ravel(), self.multi_gaussian_MLE[0]["cov"], allow_singular=True)))
        X_log_densities_positive = pd.DataFrame(X_log_densities_positive, columns = ["X_log_densities_positive"])
        X_log_densities_negative = pd.DataFrame(X_log_densities_negative, columns = ["X_log_densities_negative"])
        X_log_masses_positive = X_categorical.apply(self.log_categorical_mass_MLE, args = (1,))
        X_log_masses_negative = X_categorical.apply(self.log_categorical_mass_MLE, args = (0,))
        log_mass_class_positive = pd.DataFrame({"1" : [np.log(self.class_MLE[1])] * num_samples})
        log_mass_class_negative = pd.DataFrame({"0" : [np.log(self.class_MLE[0])] * num_samples})
        X_log_densities_positive.reset_index(drop = True, inplace = True)
        X_log_densities_negative.reset_index(drop = True, inplace = True)
        X_log_masses_positive.reset_index(drop = True, inplace = True)
        X_log_masses_negative.reset_index(drop = True, inplace = True)
        log_mass_class_positive.reset_index(drop = True, inplace = True)
        log_mass_class_negative.reset_index(drop = True, inplace = True)
        X_concat_positive = pd.concat([X_log_densities_positive, X_log_masses_positive, log_mass_class_positive], axis = 1)
        X_concat_negative = pd.concat([X_log_densities_negative, X_log_masses_negative, log_mass_class_negative], axis = 1)
        X_sum_positive = X_concat_positive.sum(axis = 1).tolist()
        X_sum_negative = X_concat_negative.sum(axis = 1).tolist()
        C = pd.DataFrame({1: X_sum_positive, 0 : X_sum_negative})
        return C 
    
    def predict_MLE_naive_assumption(self, X):
        """
        Perform classification on an array of test vectors X assuming the naive assumption among the continuous attributes using a mixed Naive Bayes classifier fit using MLE
        
        Parameters
        ----------
        X : pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors

        Returns
        -------
        C: pandas Series, shape = [num_samples,]
           a pandas Series that stores the class prediction for test vector
        """
        C = self.predict_proba_MLE_naive_assumption(X)
        C = C.idxmax(axis = 1)
        return C
    
    def predict_MLE(self, X):
        """
        Perform classification on an array of test vectors X without assuming the naive assumption among the continuous attributes using a mixed Naive Bayes classifier fit using MAP
        
        Parameters
        ----------
        X : pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors

        Returns
        -------
        C: pandas Series, shape = [num_samples,]
           a pandas Series that stores the class prediction for test vector
        """
        C = self.predict_proba_MLE(X)
        C = C.idxmax(axis = 1)
        return C
    
    def predict_proba_MAP_naive_assumption(self, X_test):
        """
        Returns probability estimates for the test vector X_test assuming the naive assumption among the continuous attributes using a mixed Naive Bayes classifier fit using MAP
        
        Parameters
        ----------
        X_test: pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors
        
        Returns
        -------
        C: pandas DataFrame, shape = [num_samples, num_classes]
            a pandas DataFrame that stores logP(x, y = 0/1| phi_MAP), values proportional to the probabilities for positive and negative
        
        """
        num_samples = X_test.shape[0]
        X_continuous = X_test[self.continuous_attributes]
        X_categorical = X_test[self.categorical_attributes]
        X_log_densities_positive = X_continuous.apply(lambda x: np.log(norm.pdf(x, loc = self.uni_gaussian_MAP[1][x.name]["mu"]  , scale = np.sqrt(1 / self.uni_gaussian_MAP[1][x.name]["precision"]))))
        X_log_densities_negative = X_continuous.apply(lambda x: np.log(norm.pdf(x, loc = self.uni_gaussian_MAP[0][x.name]["mu"]  , scale = np.sqrt(1 / self.uni_gaussian_MAP[0][x.name]["precision"]))))
        X_log_masses_positive = X_categorical.apply(self.log_categorical_mass_MAP, args = (1,))
        X_log_masses_negative = X_categorical.apply(self.log_categorical_mass_MAP, args = (0,))
        log_mass_class_positive = pd.DataFrame({"1" : [np.log(self.class_MAP[1])] * num_samples})
        log_mass_class_negative = pd.DataFrame({"0" : [np.log(self.class_MAP[0])] * num_samples})
        X_log_densities_positive.reset_index(drop = True, inplace = True)
        X_log_densities_negative.reset_index(drop = True, inplace = True)
        X_log_masses_positive.reset_index(drop = True, inplace = True)
        X_log_masses_negative.reset_index(drop = True, inplace = True)
        log_mass_class_positive.reset_index(drop = True, inplace = True)
        log_mass_class_negative.reset_index(drop = True, inplace = True)
        X_concat_positive = pd.concat([X_log_densities_positive, X_log_masses_positive, log_mass_class_positive], axis = 1)
        X_concat_negative = pd.concat([X_log_densities_negative, X_log_masses_negative, log_mass_class_negative], axis = 1)
        X_sum_positive = X_concat_positive.sum(axis = 1).tolist()
        X_sum_negative = X_concat_negative.sum(axis = 1).tolist()
        C = pd.DataFrame({1: X_sum_positive, 0 : X_sum_negative})
        return C
    
    def predict_proba_MAP(self, X_test):
        """
        Returns probability estimates for the test vector X_test without assuming the naive assumption among the continuous attributes using a mixed Naive Bayes classifier fit using MAP
        
        Parameters
        ----------
        X_test: pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors
        
        Returns
        -------
        C: pandas DataFrame, shape = [num_samples, num_classes]
            a pandas DataFrame that stores logP(x, y = 0/1| phi_MAP), values proportional to the probabilities for positive and negative
        
        """
        num_samples = X_test.shape[0]
        X_continuous = X_test[self.continuous_attributes]
        X_continuous = X_continuous.to_numpy()
        X_categorical = X_test[self.categorical_attributes]
        X_log_densities_positive = list()
        X_log_densities_negative = list()
        for record in X_continuous:
            X_log_densities_positive.append(np.log(multivariate_normal.pdf(record.ravel(), self.multi_gaussian_MAP[1]["mean"].ravel(), self.multi_gaussian_MAP[1]["cov"], allow_singular=True)))
            X_log_densities_negative.append(np.log(multivariate_normal.pdf(record.ravel(), self.multi_gaussian_MAP[0]["mean"].ravel(), self.multi_gaussian_MAP[0]["cov"], allow_singular=True)))
        X_log_densities_positive = pd.DataFrame(X_log_densities_positive, columns = ["X_log_densities_positive"])
        X_log_densities_negative = pd.DataFrame(X_log_densities_negative, columns = ["X_log_densities_negative"])
        X_log_masses_positive = X_categorical.apply(self.log_categorical_mass_MAP, args = (1,))
        X_log_masses_negative = X_categorical.apply(self.log_categorical_mass_MAP, args = (0,))
        log_mass_class_positive = pd.DataFrame({"1" : [np.log(self.class_MAP[1])] * num_samples})
        log_mass_class_negative = pd.DataFrame({"0" : [np.log(self.class_MAP[0])] * num_samples})
        X_log_densities_positive.reset_index(drop = True, inplace = True)
        X_log_densities_negative.reset_index(drop = True, inplace = True)
        X_log_masses_positive.reset_index(drop = True, inplace = True)
        X_log_masses_negative.reset_index(drop = True, inplace = True)
        log_mass_class_positive.reset_index(drop = True, inplace = True)
        log_mass_class_negative.reset_index(drop = True, inplace = True)
        X_concat_positive = pd.concat([X_log_densities_positive, X_log_masses_positive, log_mass_class_positive], axis = 1)
        X_concat_negative = pd.concat([X_log_densities_negative, X_log_masses_negative, log_mass_class_negative], axis = 1)
        X_sum_positive = X_concat_positive.sum(axis = 1).tolist()
        X_sum_negative = X_concat_negative.sum(axis = 1).tolist()
        C = pd.DataFrame({1: X_sum_positive, 0 : X_sum_negative})
        return C  

    def predict_MAP_naive_assumption(self, X):
        """
        Perform classification on an array of test vectors X assuming the naive assumption among the continuous attributes
        
        Parameters
        ----------
        X : pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors

        Returns
        -------
        C: pandas Series, shape = [num_samples,]
           a pandas Series that stores the class prediction for test vector
        """
        C = self.predict_proba_MAP_naive_assumption(X)
        C = C.idxmax(axis = 1)
        return C
    
    def predict_MAP(self, X):
        """
        Perform classification on an array of test vectors X without assuming the naive assumption among the continuous attributes
        
        Parameters
        ----------
        X : pandas DataFrame, shape = [num_samples, num_features]
            a pandas DataFrame that stores all the unseen test feature vectors

        Returns
        -------
        C: pandas Series, shape = [num_samples,]
           a pandas Series that stores the class prediction for test vector
        """
        C = self.predict_proba_MAP(X)
        C = C.idxmax(axis = 1)
        return C
    
# %% Model training and testing
mnb = MixedNB(continuous_attributes, categorical_attributes)
mnb.fit_MAP(X_train, y_train, naive_assumption = True)
y_pred = mnb.predict_MAP_naive_assumption(X_test)
accuracy = accuracy_score(y_test, y_pred)

mnb = MixedNB(continuous_attributes, categorical_attributes)
mnb.fit_MAP(X_train, y_train, naive_assumption = False)
y_pred = mnb.predict_MAP(X_test)
accuracy = accuracy_score(y_test, y_pred)

mnb = MixedNB(continuous_attributes, categorical_attributes)
mnb.fit_MLE(X_train, y_train, naive_assumption = True)
y_pred = mnb.predict_MLE_naive_assumption(X_test)
accuracy = accuracy_score(y_test, y_pred)

mnb = MixedNB(continuous_attributes, categorical_attributes)
mnb.fit_MLE(X_train, y_train, naive_assumption = False)
y_pred = mnb.predict_MLE(X_test)
accuracy = accuracy_score(y_test, y_pred)




