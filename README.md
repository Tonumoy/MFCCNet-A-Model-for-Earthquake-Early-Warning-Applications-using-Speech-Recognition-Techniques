
# MFCCNet: A Network for Detection of Earthquakes using MFCC and Filter Bank Features for EEWS applications

## General Overview
Earthquakes are one of the most devastating natural phenomenons.Earthquake signals are non-stationary in nature and thus in real time, it is difficult to identify and classify events based on classical approaches like peak ground displacement, peak ground velocity or even the widely recognized algorithm of STA/LTA as they require extensive research to determine basic thresholding parameters so as to trigger an alarm. Many times due to human error or other unavoidable natural factors such as thunder strikes or landslides, the conventional algorithms may end up raising a false alarm. This work focuses on detecting earthquakes by converting seismograph recorded data into corresponding audio signals for better perception and uses popular Speech Recognition techniques like Filter bank coefficients and MEl Frequency Cepstral Coefficients(MFCC) to extract the features out of the audio signals. These features were then used to train a Convolutional Neural Network(CNN) and a Long Short Term Memory(LSTM) network. The proposed method can overcome the above mentioned problems and help in detecting earthquakes automatically from the waveforms without much human intervention.For 1000Hz audio data set the CNN  model showed a testing accuracy of 91.102\% for 0.2 second sample  window length while the LSTM model showed 93.999\% for the same. Since the input of the method is only the waveform, it is suitable for real-time processing, thus, the models can very well be used also as an onsite earthquake early warning system requiring a minimum amount of preparation time and workload.
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/CNN_Vs_LSTM_Model_Acccuracy_1000Fs%20.png?raw=true)

## Data Description
Indian Earthquake acceleration data of the past 10 years and magnitudes approximately between 2-8 was collected from PESMOS (Program for Excellence in Strong Motion Studies, IIT Roorkee) and EGMB (Eastern Ghats Mobile Belt) Data of earthquakes with magnitudes ranging between 2-5 Collected at Geology and Geophysics Lab IIT Kharagpur, were also used. 
Before training the models, proper cleaning of the audio data was done and essential features from the data were then extracted using popular speech processing methodologies of Filter Banks and Mel-Frequency Cepstral Coefficients (MFCCs).

## Model Architectures
### CNN Model Architecture
The proposed architecture for the CNN  model has 4 convolutional layers, 1 maxpool layer and 3 dense layers. The first layer of the 4 convolutional layers consists of 16 filters built with 3x3 convolution with 'Relu' as the activation function and 1x1 stride. All the parameters for the second, third and fourth layers remain the same except for the number of filters in each layer multiplies two times with the number of filters in the previous layer. Or in other words, 16 filters in the first layer, 32 filters in the second layer, 64 in the third and 128 in the final layer. 
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/16%20Filters%20of%20Layer%201%20in%204X4.png?raw=true)
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/CNN%20Diagram.png?raw=true)

The idea behind increasing the number of filters in each layer is to be more specific about the features as the data starts to convolve through each layer. A kernel of 2x2 has been used for the maxpool layer. The three dense layers after the maxpool layer consist of 128, 64 and 2 neurons so as to pool down the features for the final 2 class classification. The first two dense layers use 'Relu' as their activation function whereas the last dense layer  uses 'Softmax' as its activation function as we use categorical cross-entropy for multi-class classification purposes and 'adam' as the optimizer.   
### LSTM Model Architecture
The proposed architecture for the LSTM  model has 2 LSTM layers consisting of 128 neurons each. 4 time distributed fully connected layers of 64, 32, 16 and 8 neurons respectively are added after the 2 LSTM layers with 'relu' as their activation function. Lastly a dense layer consisting of 2 neurons is added for the final 2 class classification with 'softmax' as its activation function and 'adam' as the optimizer. 
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/LSTM%20Diagram.png?raw=true)

## Filter Banks and Mel-Frequency Cepstral Coefficients (MFCCs)
The rational behind using filter banks was to separate the input signal into its multiple components such that each of those components carries a single frequency sub-band of the original signal. Triangular filters, typically 26, were applied on a Mel-scale to the power spectrum of the short-time frames to extract the frequency bands.
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/Filterbank64(spectral).png?raw=true)

The formula for converting from frequency to Mel scale is given by:
\begin{equation}
    M(f) = 1125 ln(1+f/700)    
\end{equation}
To go back from Mels to frequency, the formula used is given by:
\begin{equation}
    M^{-1}(m) = 700(exp (m/1125)-1)
\end{equation}

![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/mfcc_64(blues).png?raw=true)

MFCC is by far the most successful and mostly used feature in the area of speech processing. For speech signals, the mean and the variance changes continuously with time and thus it makes the signal non-stationary. Similarly like speech, earthquake signals are also non-stationary as each of them have different arrival of P, S and surface waves. Therefore, normal signal processing techniques like fourier transfrom, cannot be directly applied to it. But, if the signal is observed in very small duration window (say 25ms), the frequency content in that small duration appears to be more or less stationary. This opened up the possibility of short time processing of the earthquake sound signals. The small duration window is called a frame, discussed in section. For processing the whole sound segment, the window was moved from the beginning to end of the segment consistently with equal steps, called shift or stride. Based on the frame-size and frame-stride, it gave us M frames. Now, for each of the frames, MFCC coefficients were computed. Moreover, the filter bank energies computed were highly correlated since all the filterbanks were overlapping. This becomes a problem for most of the machine learning algorithms. To reduce autocorrelation between the filterbank coefficients and get a compressed representation of the filter banks, a Discrete Cosine Transform (DCT) was applied to the filterbank energies. This also allowed the  use of diagonal covariance matrices to model the features for training. Also, 13 of the 26 DCT coefficients were kept and the rest were discarded due to the fact that fast changes in the filterbank energies are represented by higher DCT coefficients. These fast changes resulted in degrading the model performances. Thus a small improvement was observed by dropping them.


## Results
The CNN and the LSTM models performed almost similarly for 200Hz audio data set, but significant improvements in the train-test accuracy percentages is observed for 1000Hz data set. For 1000 Hz audio data set the CNN model showed a testing accuracy of 91.102\% for 0.2 second sample window length while the LSTM model showed 93.999\% for the same. This observation can be backed by the fact that LSTMs performs better for sequential or time series data classifications. 

The Kappa statistics(values), generally used for comparing an Observed Accuracy with an Expected Accuracy (random chance), was used for validating the model accuracies for both the data sets.
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/Kappa_plots1000&200Fs.png?raw=true)

For both data classes, activations by random 5 out of 16 filters of the first layer of the CNN Model along with their inputs is represented by the following figures.
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/Positive%20Class%20Activations%20with%20corresponding%20Filters.png?raw=true)

![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/Negative%20Class%20Activations%20with%20corresponding%20Filters.png?raw=true)

![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/histograms.png?raw=true)

This is one of the very first applications of Machine learning in the field of earthquake detection using MFCC's and Filterbank Coefficients which are generally used in the field of Speech Recognition. An earthquake is basically understood well by three types of waves namely P-wave, S-wave and surface wave. Interaction of these waves with the surrounding medium gives an earthquake its intensity. Any wave is a vibration. Any vibration has some sound associated with it. It might be inaudible to the human ears, but the sound remains. In real time, it is difficult to identify and classify events based on classical approaches like peak ground displacement, peak ground velocity or even the widely recognized algorithm of STA/LTA as they require extensive research to determine basic thresholding parameters so as to trigger an alarm. Many times due to human error or other unavoidable natural factors such as thunder strikes or landslides, the conventional algorithms may end up raising a false alarm. 

![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/TimeSeriesVsSpectrogram%201.jpg?raw=true)

 The proposed method can overcome these problems as it can extract essential features of a raw sound or vibrational data without manually hand engineering them and doesn’t require any knowledge or expertise in the relevant field. It is also invariant to small variations in occurrence in time or position and can understand representations of data with multiple levels of abstraction. But the most interesting and effective part of this model is that, it can be trained on all these various classes of sounds other than earthquake sounds to classify every signal that the sensor detects, using their sound signatures. This can be of enormous help in military applications also. If trained with human movement sounds, the model could be deployed in the border and other high security areas so as to provide us instant information regarding trespassing and other such unlawful activities, releasing the burden to some extent from the security officials and the soldiers. This could be a solution not just for earthquake detection alone but for many other such applications also.

## Contributors
* Tonumoy Mukherjee tonumoymukherjee2@gmail.com

## License & Copyright
&#169; Tonumoy Mukherjee, Indian Insitute of Technology Kharagpur
> Licensed under the [MIT License](LICENSE).
