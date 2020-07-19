<<<<<<< HEAD

#%% Headers
import os
from tqdm import tqdm #used to wrap any iterable, creates a progress bar to visualize any loop in execution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa #default audio library for handling audio files
#import pdb
import scipy
from matplotlib import cm
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
#from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
import pickle
from keras.callbacks import ModelCheckpoint
#from cfg import Config
import random
#import theano
from keras.utils import plot_model
#import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

#%% Data Processing and Cleaning

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
        


#%% Configuration

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=256, rate=1000):
        self.mode = mode
        self.nfilt= nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10) #10 th of a second
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        
#%% Model
        
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading exixting data for {} model' .format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
# Feature Extraction
        
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean_train/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample) 
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y  = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=2)
    config.data = (X, y)
    
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    
    return X,y

# CNN Model
    
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    #pdb.set_trace()
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), 
                     
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    #adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['acc'])
    #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    return model

# LSTM Model
    
def get_recurrent_model():
    #shape of data for RNN is (n, time, features)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.summary()
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['acc'])
#    model.contrib.layers.l2_regularizer(
#    scale=1 ,
#    scope=None
#)
    return model

# Data Management & Model Selection

df = pd.read_csv('Quake_mod.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean_train/'+f)
    signal =signal[0:int(0.2*rate)] #first 0.2 sec of signal
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1) #10th of a second
prob_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08,fontsize='large', fontweight='bold')
ax.pie(class_dist, labels=class_dist.index,autopct='%2.2f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode='conv')

if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
    
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

#   Training 
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=200, batch_size=32, shuffle=True, 
          class_weight=class_weight, validation_split=0.1,
          callbacks=[checkpoint])

model.save(config.model_path)
plot_model(model, to_file='convolutional_neural_network.png')   
    
#%%   

#def plot_filters(layer,X,y):
##    
##    
##    filters = layer.W.get_value()
#    filters, biases = layer.get_weights()
#    fig = plt.figure()
#    for j in range(len (filters)):
#        ax = fig.add_subplot(y,X,j+1)
#        ax.matshow(filters[j][0], cmap = cm.binary)
##        
#        plt.xticks(np.array([]))
#        plt.yticks(np.array([]))
#    plt.tight_layout()
#    return plt
##
#plot_filters(model.layers[0],4,4) #first convolution layer filters
##    
##%%
#for layer in model.layers:
#	# check for convolutional layer
#	if 'conv' not in layer.name:
#		continue
#	# get filter weights
#	filters, biases = layer.get_weights()
#	print(layer.name, filters.shape)
    
# Apda Cde Img Resize Nearest Neighbour
    
def my_resize(arr, f):
    newarr = np.ones((arr.shape[0]*f, arr.shape[1]*f, arr.shape[2], arr.shape[3]))
    for k1 in range(arr.shape[2]):
        for k2 in range(arr.shape[3]):
            temp = arr[:, :, k1, k2]
            temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    newarr[i*f:(i+1)*f, j*f:(j+1)*f, k1, k2]=temp[i, j]
    return newarr

def plot_filter(arr, f, padd):
    up_arr = my_resize(arr, f)
    newarr = np.ones((arr.shape[2]*(up_arr.shape[0]+padd), arr.shape[3]*(up_arr.shape[1]+padd)))  
    for i in range(arr.shape[2]):
        for j in range(arr.shape[3]):
            newarr[i*up_arr.shape[0]+i*padd:(i+1)*up_arr.shape[0]+i*padd, j*up_arr.shape[0]+j*padd:(j+1)*up_arr.shape[0]+j*padd]= \
            up_arr[:,:,i, j]
    return newarr

#    Filter output plots CNN

fig1, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4 , ncols=1)
ax1.set_title("Layer 1 - 16 Filters")
#ax1.set_xlabel("X-label for axis 1"   
filters, bias = model.layers[0].get_weights() #1st layer 16 filters
#filters = filters.reshape(3, 3, 4,4)
#title_obj = plt.title('16 Filters of Layer - 1') #get the title property handler                           
#plt.getp(title_obj, 'text') #print out the properties of title
out = plot_filter(filters, 8, 1)
ax1.imshow(out, cmap=cm.gray)   
 
filters, bias = model.layers[1].get_weights() #2nd layer 32 filters   
out = random.sample(list(plot_filter(filters, 8, 1)),32)    
ax2.imshow(out, cmap=cm.gray)
ax2.set_title("Layer 2 - 16 X 32 Filters")    

filters, bias = model.layers[2].get_weights() #3rd layer 64 filters   
out = random.sample(list(plot_filter(filters, 8, 1)),64)    
ax3.imshow(out, cmap=cm.gray)
ax3.set_title("Layer 3 - 32 X 64 Filters")  

filters, bias = model.layers[3].get_weights() #4thlayer 128 filters    
out = random.sample(list(plot_filter(filters, 8, 1)),128)    
ax4.imshow(out, cmap=cm.gray)      
ax4.set_title("Layer 4 - 64 X 128 Filters")
       
    
fig2, axs = plt.subplots(nrows=2 , ncols=5)
axs[0,0].imshow(X[1,:,:,0]) #Positive Class I/P
axs[0,0].set_title("Positive Class I/P")
axs[1,0].imshow(X[0,:,:,0]) #Negative Class I/P
axs[1,0].set_title("Negative Class I/P")

axs[0,1].imshow(X[5,:,:,0]) #Positive Class I/P
axs[0,1].set_title("Positive Class I/P")
axs[1,1].imshow(X[6,:,:,0]) #Negative Class I/P
axs[1,1].set_title("Negative Class I/P")

axs[0,2].imshow(X[8,:,:,0]) #Positive Class I/P
axs[0,2].set_title("Positive Class I/P")
axs[1,2].imshow(X[9,:,:,0]) #Negative Class I/P
axs[1,2].set_title("Negative Class I/P")

axs[0,3].imshow(X[20,:,:,0]) #Positive Class I/P
axs[0,3].set_title("Positive Class I/P")
axs[1,3].imshow(X[21,:,:,0]) #Negative Class I/P
axs[1,3].set_title("Negative Class I/P")

axs[0,4].imshow(X[24,:,:,0]) #Positive Class I/P
axs[0,4].set_title("Positive Class I/P")
axs[1,4].imshow(X[25,:,:,0]) #Negative Class I/P
axs[1,4].set_title("Negative Class I/P")


#from keras import backend as K
#def get_activations(model, layer_idx, X_batch):
#    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
#    activations = get_activations([X_batch,0])
#    return activations


# visualizing intermediate layers

#output_layer = model.layers[0].get_output()
#output_fn = theano.function([model.layers[0].get_input()], output_layer)
#
## the input image
#
#input_image=X[1,:,:,0]
#print(input_image.shape)
#
#plt.imshow(input_image[0,:,:,0],cmap ='gray')
#plt.imshow(input_image[0,0,:,0])
#
#
#output_image = output_fn(input_image)
#print(output_image.shape)
#
## Rearrange dimension so we can plot the result 
#output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
#print(output_image.shape)


fig3, axs = plt.subplots(nrows=3 , ncols=5)

filters, bias = model.layers[3].get_weights()

filt1 = filters[:,:,0,0] # 1st filter
filt2 = filters[:,:,0,1] # 2nd filter
filt3 = filters[:,:,0,11] # 3rd filter
filt4 = filters[:,:,0,13] # 4th filter
filt5 = filters[:,:,0,14] # 5th filter

inp1 = X[8,:,:,0] # random input

fst_conv = scipy.signal.convolve2d(inp1, filt1, mode='same', boundary='fill', fillvalue=0) #first filter convolution
fst_conv[fst_conv<0] = 0 #relu

scnd_conv = scipy.signal.convolve2d(inp1, filt2, mode='same', boundary='fill', fillvalue=0) #second filter convolution
scnd_conv[scnd_conv<0] = 0 #relu

thrd_conv = scipy.signal.convolve2d(inp1, filt3, mode='same', boundary='fill', fillvalue=0) #third filter convolution
thrd_conv[thrd_conv<0] = 0 #relu

frth_conv = scipy.signal.convolve2d(inp1, filt4, mode='same', boundary='fill', fillvalue=0) #fourth filter convolution
frth_conv[frth_conv<0] = 0 #relu

ffth_conv = scipy.signal.convolve2d(inp1, filt5, mode='same', boundary='fill', fillvalue=0) #fifth filter convolution
ffth_conv[ffth_conv<0] = 0 #relu 


axs[0,0].imshow(filt1, cmap =cm.gray)
axs[0,0].set_title("Layer 1, Filter 1")

axs[0,1].imshow(filt2, cmap =cm.gray)
axs[0,1].set_title("Layer 1, Filter 2")

axs[0,2].imshow(filt3, cmap =cm.gray)
axs[0,2].set_title("Layer 1, Filter 3")

axs[0,3].imshow(filt4, cmap =cm.gray)
axs[0,3].set_title("Layer 1, Filter 4")

axs[0,4].imshow(filt5, cmap =cm.gray)
axs[0,4].set_title("Layer 1, Filter 5")




axs[1,0].imshow(inp1, cmap =cm.gray)
axs[1,1].imshow(inp1, cmap =cm.gray)
axs[1,2].imshow(inp1, cmap =cm.gray)
axs[1,2].set_title("Identical Positive Input to the filters")
axs[1,3].imshow(inp1, cmap =cm.gray)
im5  = axs[1,4].imshow(inp1, cmap =cm.gray)
divider = make_axes_locatable(axs[1,4])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im5, cax=cax, orientation='vertical')



axs[2,0].imshow(fst_conv, cmap =cm.gray)
axs[2,0].set_title("Layer 1, Filter 1 Activation")

axs[2,1].imshow(scnd_conv, cmap =cm.gray)
axs[2,1].set_title("Layer 1, Filter 2 Activation")

axs[2,2].imshow(thrd_conv, cmap =cm.gray)
axs[2,2].set_title("Layer 1, Filter 3 Activation")

axs[2,3].imshow(frth_conv, cmap =cm.gray)
axs[2,3].set_title("Layer 1, Filter 4 Activation")

axs[2,4].imshow(ffth_conv, cmap =cm.gray)
axs[2,4].set_title("Layer 1, Filter 5 Activation")

#plt.imshow(conv, cmap = cm.gray) # activations




## Predict

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extacting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min) 
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
        
    return y_true, y_pred, fn_prob
    
    
df = pd.read_csv('test_new.csv') # test csv file input
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles' , 'time.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean_test') #test data folder input
acc_score = accuracy_score(y_true = y_true, y_pred = y_pred)

kappa_score = cohen_kappa_score(y_true, y_pred, labels=None, weights=None, sample_weight=None)
cfn_mat = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None) #confusion matrix
y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p
        
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions_time_1000fs 0.2 sec win.csv', index = False) #prediction output


=======

#%% Headers
import os
from tqdm import tqdm #used to wrap any iterable, creates a progress bar to visualize any loop in execution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa #default audio library for handling audio files
#import pdb
import scipy
from matplotlib import cm
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
#from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
import pickle
from keras.callbacks import ModelCheckpoint
#from cfg import Config
import random
#import theano
from keras.utils import plot_model
#import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

#%% Data Processing and Cleaning

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
        


#%% Configuration

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=256, rate=1000):
        self.mode = mode
        self.nfilt= nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10) #10 th of a second
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        
#%% Model
        
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading exixting data for {} model' .format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
# Feature Extraction
        
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean_train/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample) 
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y  = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=2)
    config.data = (X, y)
    
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    
    return X,y

# CNN Model
    
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    #pdb.set_trace()
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), 
                     
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    #adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['acc'])
    #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    return model

# LSTM Model
    
def get_recurrent_model():
    #shape of data for RNN is (n, time, features)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.summary()
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['acc'])
#    model.contrib.layers.l2_regularizer(
#    scale=1 ,
#    scope=None
#)
    return model

# Data Management & Model Selection

df = pd.read_csv('Quake_mod.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean_train/'+f)
    signal =signal[0:int(0.2*rate)] #first 0.2 sec of signal
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1) #10th of a second
prob_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08,fontsize='large', fontweight='bold')
ax.pie(class_dist, labels=class_dist.index,autopct='%2.2f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode='conv')

if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
    
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

#   Training 
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=200, batch_size=32, shuffle=True, 
          class_weight=class_weight, validation_split=0.1,
          callbacks=[checkpoint])

model.save(config.model_path)
plot_model(model, to_file='convolutional_neural_network.png')   
    
#%%   

#def plot_filters(layer,X,y):
##    
##    
##    filters = layer.W.get_value()
#    filters, biases = layer.get_weights()
#    fig = plt.figure()
#    for j in range(len (filters)):
#        ax = fig.add_subplot(y,X,j+1)
#        ax.matshow(filters[j][0], cmap = cm.binary)
##        
#        plt.xticks(np.array([]))
#        plt.yticks(np.array([]))
#    plt.tight_layout()
#    return plt
##
#plot_filters(model.layers[0],4,4) #first convolution layer filters
##    
##%%
#for layer in model.layers:
#	# check for convolutional layer
#	if 'conv' not in layer.name:
#		continue
#	# get filter weights
#	filters, biases = layer.get_weights()
#	print(layer.name, filters.shape)
    
# Apda Cde Img Resize Nearest Neighbour
    
def my_resize(arr, f):
    newarr = np.ones((arr.shape[0]*f, arr.shape[1]*f, arr.shape[2], arr.shape[3]))
    for k1 in range(arr.shape[2]):
        for k2 in range(arr.shape[3]):
            temp = arr[:, :, k1, k2]
            temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    newarr[i*f:(i+1)*f, j*f:(j+1)*f, k1, k2]=temp[i, j]
    return newarr

def plot_filter(arr, f, padd):
    up_arr = my_resize(arr, f)
    newarr = np.ones((arr.shape[2]*(up_arr.shape[0]+padd), arr.shape[3]*(up_arr.shape[1]+padd)))  
    for i in range(arr.shape[2]):
        for j in range(arr.shape[3]):
            newarr[i*up_arr.shape[0]+i*padd:(i+1)*up_arr.shape[0]+i*padd, j*up_arr.shape[0]+j*padd:(j+1)*up_arr.shape[0]+j*padd]= \
            up_arr[:,:,i, j]
    return newarr

#    Filter output plots CNN

fig1, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4 , ncols=1)
ax1.set_title("Layer 1 - 16 Filters")
#ax1.set_xlabel("X-label for axis 1"   
filters, bias = model.layers[0].get_weights() #1st layer 16 filters
#filters = filters.reshape(3, 3, 4,4)
#title_obj = plt.title('16 Filters of Layer - 1') #get the title property handler                           
#plt.getp(title_obj, 'text') #print out the properties of title
out = plot_filter(filters, 8, 1)
ax1.imshow(out, cmap=cm.gray)   
 
filters, bias = model.layers[1].get_weights() #2nd layer 32 filters   
out = random.sample(list(plot_filter(filters, 8, 1)),32)    
ax2.imshow(out, cmap=cm.gray)
ax2.set_title("Layer 2 - 16 X 32 Filters")    

filters, bias = model.layers[2].get_weights() #3rd layer 64 filters   
out = random.sample(list(plot_filter(filters, 8, 1)),64)    
ax3.imshow(out, cmap=cm.gray)
ax3.set_title("Layer 3 - 32 X 64 Filters")  

filters, bias = model.layers[3].get_weights() #4thlayer 128 filters    
out = random.sample(list(plot_filter(filters, 8, 1)),128)    
ax4.imshow(out, cmap=cm.gray)      
ax4.set_title("Layer 4 - 64 X 128 Filters")
       
    
fig2, axs = plt.subplots(nrows=2 , ncols=5)
axs[0,0].imshow(X[1,:,:,0]) #Positive Class I/P
axs[0,0].set_title("Positive Class I/P")
axs[1,0].imshow(X[0,:,:,0]) #Negative Class I/P
axs[1,0].set_title("Negative Class I/P")

axs[0,1].imshow(X[5,:,:,0]) #Positive Class I/P
axs[0,1].set_title("Positive Class I/P")
axs[1,1].imshow(X[6,:,:,0]) #Negative Class I/P
axs[1,1].set_title("Negative Class I/P")

axs[0,2].imshow(X[8,:,:,0]) #Positive Class I/P
axs[0,2].set_title("Positive Class I/P")
axs[1,2].imshow(X[9,:,:,0]) #Negative Class I/P
axs[1,2].set_title("Negative Class I/P")

axs[0,3].imshow(X[20,:,:,0]) #Positive Class I/P
axs[0,3].set_title("Positive Class I/P")
axs[1,3].imshow(X[21,:,:,0]) #Negative Class I/P
axs[1,3].set_title("Negative Class I/P")

axs[0,4].imshow(X[24,:,:,0]) #Positive Class I/P
axs[0,4].set_title("Positive Class I/P")
axs[1,4].imshow(X[25,:,:,0]) #Negative Class I/P
axs[1,4].set_title("Negative Class I/P")


#from keras import backend as K
#def get_activations(model, layer_idx, X_batch):
#    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
#    activations = get_activations([X_batch,0])
#    return activations


# visualizing intermediate layers

#output_layer = model.layers[0].get_output()
#output_fn = theano.function([model.layers[0].get_input()], output_layer)
#
## the input image
#
#input_image=X[1,:,:,0]
#print(input_image.shape)
#
#plt.imshow(input_image[0,:,:,0],cmap ='gray')
#plt.imshow(input_image[0,0,:,0])
#
#
#output_image = output_fn(input_image)
#print(output_image.shape)
#
## Rearrange dimension so we can plot the result 
#output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
#print(output_image.shape)


fig3, axs = plt.subplots(nrows=3 , ncols=5)

filters, bias = model.layers[3].get_weights()

filt1 = filters[:,:,0,0] # 1st filter
filt2 = filters[:,:,0,1] # 2nd filter
filt3 = filters[:,:,0,11] # 3rd filter
filt4 = filters[:,:,0,13] # 4th filter
filt5 = filters[:,:,0,14] # 5th filter

inp1 = X[8,:,:,0] # random input

fst_conv = scipy.signal.convolve2d(inp1, filt1, mode='same', boundary='fill', fillvalue=0) #first filter convolution
fst_conv[fst_conv<0] = 0 #relu

scnd_conv = scipy.signal.convolve2d(inp1, filt2, mode='same', boundary='fill', fillvalue=0) #second filter convolution
scnd_conv[scnd_conv<0] = 0 #relu

thrd_conv = scipy.signal.convolve2d(inp1, filt3, mode='same', boundary='fill', fillvalue=0) #third filter convolution
thrd_conv[thrd_conv<0] = 0 #relu

frth_conv = scipy.signal.convolve2d(inp1, filt4, mode='same', boundary='fill', fillvalue=0) #fourth filter convolution
frth_conv[frth_conv<0] = 0 #relu

ffth_conv = scipy.signal.convolve2d(inp1, filt5, mode='same', boundary='fill', fillvalue=0) #fifth filter convolution
ffth_conv[ffth_conv<0] = 0 #relu 


axs[0,0].imshow(filt1, cmap =cm.gray)
axs[0,0].set_title("Layer 1, Filter 1")

axs[0,1].imshow(filt2, cmap =cm.gray)
axs[0,1].set_title("Layer 1, Filter 2")

axs[0,2].imshow(filt3, cmap =cm.gray)
axs[0,2].set_title("Layer 1, Filter 3")

axs[0,3].imshow(filt4, cmap =cm.gray)
axs[0,3].set_title("Layer 1, Filter 4")

axs[0,4].imshow(filt5, cmap =cm.gray)
axs[0,4].set_title("Layer 1, Filter 5")




axs[1,0].imshow(inp1, cmap =cm.gray)
axs[1,1].imshow(inp1, cmap =cm.gray)
axs[1,2].imshow(inp1, cmap =cm.gray)
axs[1,2].set_title("Identical Positive Input to the filters")
axs[1,3].imshow(inp1, cmap =cm.gray)
im5  = axs[1,4].imshow(inp1, cmap =cm.gray)
divider = make_axes_locatable(axs[1,4])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im5, cax=cax, orientation='vertical')



axs[2,0].imshow(fst_conv, cmap =cm.gray)
axs[2,0].set_title("Layer 1, Filter 1 Activation")

axs[2,1].imshow(scnd_conv, cmap =cm.gray)
axs[2,1].set_title("Layer 1, Filter 2 Activation")

axs[2,2].imshow(thrd_conv, cmap =cm.gray)
axs[2,2].set_title("Layer 1, Filter 3 Activation")

axs[2,3].imshow(frth_conv, cmap =cm.gray)
axs[2,3].set_title("Layer 1, Filter 4 Activation")

axs[2,4].imshow(ffth_conv, cmap =cm.gray)
axs[2,4].set_title("Layer 1, Filter 5 Activation")

#plt.imshow(conv, cmap = cm.gray) # activations




## Predict

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extacting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min) 
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
        
    return y_true, y_pred, fn_prob
    
    
df = pd.read_csv('test_new.csv') # test csv file input
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles' , 'time.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean_test') #test data folder input
acc_score = accuracy_score(y_true = y_true, y_pred = y_pred)

kappa_score = cohen_kappa_score(y_true, y_pred, labels=None, weights=None, sample_weight=None)
cfn_mat = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None) #confusion matrix
y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p
        
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions_time_1000fs 0.2 sec win.csv', index = False) #prediction output


>>>>>>> 054d757... Initial
