
# MFCCNet: A Network for Detection of Earthquakes using MFCC and Filter Bank Features for EEWS applications

## General Overview
This work focuses on detecting earthquakes by converting seismograph recorded data into corresponding audio signals for better perception and uses popular Speech Recognition techniques like Filter bank coefficients and MEl Frequency Cepstral Coefficients(MFCC) to extract the features out of the audio signals. These features were then used to train a Convolutional Neural Network(CNN) and a Long Short Term Memory(LSTM) network. The proposed method can help in detecting earthquakes automatically from the waveforms without much human intervention. Since the input of the method is only the waveform, it is suitable for real-time processing, thus, the models can very well be used also as an onsite earthquake early warning system requiring a minimum amount of preparation time and workload.
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/Results/Accurcy_plots1000&200Fs.png?raw=true)

## Data Description
Indian Earthquake acceleration data of the past 10 years and magnitudes approximately between 2-8 was collected from PESMOS (Program for Excellence in Strong Motion Studies, IIT Roorkee) and EGMB (Eastern Ghats Mobile Belt) Data of earthquakes with magnitudes ranging between 2-5 Collected at Geology and Geophysics Lab IIT Kharagpur, were also used. 
Before training the models, proper cleaning of the audio data was done and essential features from the data were then extracted using popular speech processing methodologies of Filter Banks and Mel-Frequency Cepstral Coefficients (MFCCs).

## Repository Contents
* 4 python files (cfg, Data Processing and Cleaning, Model, Predict)
* Train Data Folder named 'clean_train'
* Test Data Folder named 'clean_test'
* Folder named 'wavfiles'containing uncleaned mix of all train and test data
* Results achived after successful code execution

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
