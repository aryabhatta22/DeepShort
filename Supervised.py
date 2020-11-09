import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

class Supervised:
    
    def __init__(self, X, y, split = True, split_ratio = 0.2):
        if split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = split_ratio, random_state = 0)
        else:
            self.X_train = X
            self.y_train = y
            self.X_test = None
            self.y_test = None
    

class Classification(Supervised):
    
    def __init__(self,X, y, split = True, split_ratio = 0.2):
        Supervised.__init__(self,X, y, split, split_ratio)
        self.LR = None
        self.DTC = None
        self.RFC = None
        self.GNB = None
    
    def  fit():
        """

        Acronyms
        ----------
        LR : Logistic Regression
        DTC : Decision Tree Classifier
        RFC : Random Forest Classifier
        GNB : Gaussian Naive Bayes

        Returns
        -------
        
        None

        """
        self.LR = LogisticRegression(random_state=0).fit(X_train, y_train)
        self.DTC = DecisionTreeClassifier().fit(X_train,y_train)
        self.RFC = RandomForestClassifier(max_depth= None, random_state=0).fit(X_train,y_train)
        self.GNB = GaussianNB().fit(X_train,y_train)
    
    def evaluate():
        if(self.X_test != None):
            lr_eval = self.LR.evaluate(X_test, y_test)
            dtc_eval = self.DTC.evaluate(X_test, y_test)
            rfc_eval = self.RFC.evaluate(X_test, y_test)
            gnb_eval = self.GNB.evaluate(X_test, y_test)
            
        