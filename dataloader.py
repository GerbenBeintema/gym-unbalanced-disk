from os import getcwd, path
from pandas import read_csv

class DATA():
    def __init__(self):
        self._get_data()

        self.train = read_csv(self.trainfilepath)
        self.testsub = read_csv(self.testsubfilepath)
        self.testsim = read_csv(self.testsimfilepath)

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