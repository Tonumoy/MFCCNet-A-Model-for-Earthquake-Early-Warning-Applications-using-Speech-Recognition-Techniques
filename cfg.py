<<<<<<< HEAD
import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=256, rate=200):
        self.mode = mode
        self.nfilt= nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10) #10 th of a second
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
=======
import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=256, rate=200):
        self.mode = mode
        self.nfilt= nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10) #10 th of a second
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
>>>>>>> 054d757... Initial
        