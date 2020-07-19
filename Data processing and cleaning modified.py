#Developed By: Tonumoy Mukherjee

import os
from tqdm import tqdm #used to wrap any iterable, creates a progress bar to visualize any loop in execution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa #default audio library for handling audio files
#import pdb

#function for plotting signals
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    #fig.set_xlabel('Time(s)')
    #fig.set_ylabel('Amplitude')
    axes = axes.reshape(2,1)
    fig.suptitle('Time Series', size=16)
    
    i = 0
    for x in range(2):
        for y in range(1):
            #pdb.set_trace()
            axes[x,y].set_title(list(signals.keys())[i], size=9)
            axes[x,y].plot(list(signals.values())[i])
            #pdb.set_trace()
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].set_xlabel('time (s)',fontsize='large',fontweight='bold')
            axes[x,y].set_ylabel('Amplitude',fontsize='large',fontweight='bold')
           
            i += 1
            
#function for plotting the Fourier Transform
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    axes = axes.reshape(2,1)
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(1):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].set_xlabel('time (s)',fontsize='large',fontweight='bold')
            axes[x,y].set_ylabel('Amplitude',fontsize='large',fontweight='bold')
            i += 1

#function for plotting the Filter Bank Coefficients
def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    axes = axes.reshape(2,1)
    #fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(1):
            axes[x,y].set_title(list(fbank.keys())[i],fontsize='18',fontweight='bold')
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='Spectral', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].set_xlabel('time (s)',fontsize='18', fontweight='bold')
            axes[x,y].set_ylabel('Frequency(KHz)',fontsize='18', fontweight='bold')
            i += 1

#function for the envelope function to detect blank spaces in the signal and 
#removing them with respect to a particular threshold
def envelope(y,rate,threshold):
    mask =[]
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold            :
            mask.append(True)
        else:
            mask.append(False)
    return mask
    
#function for calulating the FFT of the I/P signal            
def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n) #normalizing the frequency magnitude(Y)
    return(Y,freq)    


#function for plotting the Mel Frequency Cepstrum Coefficients (MFCC)
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    axes = axes.reshape(2,1)
    #fig.suptitle('Mel Frequency Cepstrum Coefficients', fontweight='bold',size=16)
    i = 0
    for x in range(2):
        for y in range(1):
            axes[x,y].set_title(list(mfccs.keys())[i],fontsize='20',fontweight='bold')
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='Blues_r', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].set_xlabel('time (s)',fontsize='20', fontweight='bold')
            axes[x,y].set_ylabel('MFCC Coefficients',fontsize='20', fontweight='bold')
            i += 1


df = pd.read_csv('Quake.csv')
df.set_index('fname', inplace = True)

for f in df.index:
    rate,signal = wavfile.read('wavfiles/'+f)
    df.at[f,'length'] = signal.shape[0]/rate
    
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08 )
ax.pie(class_dist, labels = class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle =90 ) 
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank =  {}
mfccs = {}


for c in classes:
    wav_file = df[df.label == c].iloc[0,0]
    signal, rate = librosa.load('wavfiles/'+wav_file, sr=1000)
    mask = envelope(signal, rate, 0.02)
    signal = signal[mask]
    #signal = signal[0:int(10 * rate)]  # Keep the first 3.5 seconds
    #signal = signal.shape[0]/rate
    #print(c)
    signals[c] = signal
    fft[c] = calc_fft(signal,rate)
    
    bank = logfbank(signal[:rate],rate, nfilt=26, nfft=256).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=256).T
    mfccs[c] = mel
    
    
plot_signals(signals)
#plt.xlabel('Time(s)', size=7)
#plt.ylabel('Amplitude', size=7)
plt.show()
    
plot_fft(fft)
plt.show()
    
plot_fbank(fbank)
plt.show()
     
plot_mfccs(mfccs)
plt.show()

if len(os.listdir('clean_train1')) == 0:
    for f in tqdm(df.fname):
        signal,rate = librosa.load('wavfiles/'+f, sr=1000)
        mask = envelope(signal, rate, 0.02)
        wavfile.write(filename ='clean_train1/'+f, rate=rate, data = signal[mask])
        
