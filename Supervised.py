import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    
    def  fit(X,y):
        
            