from os import getcwd, path
from pandas import read_csv, DataFrame
from numpy import array, concatenate

class DATA():
    def __init__(self):
        self._get_data()

        self.train = self._rename_df(read_csv(self.trainfilepath))
        self.testsub = self._rename_df(read_csv(self.testsubfilepath))
        self.testsim = read_csv(self.testsimfilepath)

        self.Xtrain, self.Ytrain = self.make_training_data(self.train.u, self.train.th, 15, 15)
        #self.testsub_data = self.make_training_data(self.testsub.u, self.testsub.th, 15, 15)

    def get_train(self):
        return self.train

    def get_testsub(self):
        return self.testsub

    def get_testsim(self):
        return self.testsim    

    def _get_data(self):
        DATA_DIR = path.join(getcwd(),'disc-benchmark-files\\')
    
        self.trainfilepath   = path.join(DATA_DIR,'training-data.csv')
        self.testsubfilepath = path.join(DATA_DIR,'test-prediction-submission-file.csv')
        self.testsimfilepath = path.join(DATA_DIR,'test-simulation-submission-file.csv')
    
    def _get_data_transform(self, na, nb):
        """Transforms the data to the correct format for the model f(u[k-15]-u[k-1],y[k-15]-y[k-1]) and returns X and Y data"""


        self.Xtrain, self.Ytrain = self.make_training_data(self.train)
        self.Xtestsim, self.Ytestsim = self.make_training_data(self.testsim)
        self.Xtestsub, self.Ytestsub = self.testsub , self.testsub[' y[k-0]'].iloc[1:len(self.testsub)]


    def make_training_data(self, ulist:list, ylist:list, na:int, nb:int):
        Xdata = []
        Ydata = []

        #create column names
        column_names=[f'u[k-{nb-i}]' for i in range(nb)]
        column_names.extend([f'y[k-{na-i}]' for i in range(na)])

        #for loop over the data:
        for k in range(max(na,nb),len(ulist)): #skip the first few indexes
            Xdata.append(concatenate([ulist[k-nb:k],ylist[k-na:k]])) 
            Ydata.append(ylist[k]) 

        return DataFrame(Xdata,columns=column_names), DataFrame(Ydata, columns=['y[k]'])


    def _rename_df(self, df):
        """Rename the dataframe colums to something more readable/callable"""
        return df.rename(columns={'# u':'u', ' th':'th'})