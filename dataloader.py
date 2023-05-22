from os import getcwd, path
from pandas import read_csv, DataFrame
from numpy import array, concatenate

class DATA():
    def __init__(self, na:int=15, nb:int=15):
        """
        Initialize the data
        Parameters:
            na: number of past y values used
            nb: number of past u values used
        returns:
            all in the class
        """


        self._get_data()

        self.train = self._rename_df(read_csv(self.trainfilepath))
        self.testsub = self._rename_df(read_csv(self.testsubfilepath))
        self.testsim = read_csv(self.testsimfilepath)

        self.Xtrain, self.Ytrain = self.make_training_data(self.train.u, self.train.th, na, nb)
        #self.testsub_data = self.make_training_data(self.testsub.u, self.testsub.th, na, nb)



    def _get_data(self):
        """Get the data from the disc-benchmark-files folder"""

        DATA_DIR = path.join(getcwd(),'disc-benchmark-files\\')
    
        self.trainfilepath   = path.join(DATA_DIR,'training-data.csv')
        self.testsubfilepath = path.join(DATA_DIR,'test-prediction-submission-file.csv')
        self.testsimfilepath = path.join(DATA_DIR,'test-simulation-submission-file.csv')
    
    def _get_data_transform(self, na, nb):
        """Transforms the data to the correct format for the model f(u[k-15]-u[k-1],y[k-15]-y[k-1]) and returns X and Y data"""
        self.Xtrain, self.Ytrain = self.make_training_data(self.train)
        self.Xtestsim, self.Ytestsim = self.make_training_data(self.testsim)
        self.Xtestsub, self.Ytestsub = self.testsub.iloc[:, 0:-1] , self.testsub[:,-1]


    def make_training_data(self, ulist:list, ylist:list, na:int, nb:int):
        """Training Data Function (only needs to be called for training data and testsim)"""
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


    def _rename_df1(self, df):
        """Rename the dataframe colums to something more callable"""
        return df.rename(columns={'# u':'u', ' th':'th'})
    
    # def _rename_df2(self, df):
    #     """Rename the dataframe colums to something more callable"""
    #     return df.rename(columns=['# u[k-15]':'u[k-15]', ' u[k-14]':'u[k-14]', ' u[k-13]':'u[k-13]', ' u[k-12]':'u[k-12]', ' u[k-11]':'u[k-11]', ' u[k-10]':'u[k-10]',' u[k-9]':'u[k-9]', ' u[k-8]':'u[k-8]',' u[k-7]':'u[k-7]', ' u[k-6]':'u[k-6]', ' u[k-5]':'u[k-5]', ' u[k-4]':'u[k-4]',' u[k-3]':'u[k-3]', ' u[k-2]':'u[k-2]', ' u[k-1]':'u[k-1]','y[k-15]':'y[k-15]', ' y[k-14]':'y[k-14]', ' y[k-13]':'y[k-13]',' y[k-12]':'y[k-12]',' y[k-11]':'y[k-11]', ' y[k-10]':'y[k-10]', ' y[k-9]':'y[k-9]', ' y[k-8]':'y[k-8]', ' y[k-7]':'y[k-7]',' y[k-6]':'y[k-6]', ' y[k-5]':'y[k-5]', ' y[k-4]':'y[k-4]', ' y[k-3]':'y[k-3]', ' y[k-2]':'y[k-2]', ' y[k-1]':'y[k-1]',' y[k-0]':'y[k-0]'])