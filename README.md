
# MFCCNet: A Network for Detection of Earthquakes using MFCC and Filter Bank Features for EEWS applications

## General Overview
Earthquakes are one of the most devastating natural phenomenons.Earthquake signals are non-stationary in nature and thus in real time, it is difficult to identify and classify events based on classical approaches like peak ground displacement, peak ground velocity or even the widely recognized algorithm of STA/LTA as they require extensive research to determine basic thresholding parameters so as to trigger an alarm. Many times due to human error or other unavoidable natural factors such as thunder strikes or landslides, the conventional algorithms may end up raising a false alarm. This work focuses on detecting earthquakes by converting seismograph recorded data into corresponding audio signals for better perception and uses popular Speech Recognition techniques like Filter bank coefficients and MEl Frequency Cepstral Coefficients(MFCC) to extract the features out of the audio signals. These features were then used to train a Convolutional Neural Network(CNN) and a Long Short Term Memory(LSTM) network. The proposed method can overcome the above mentioned problems and help in detecting earthquakes automatically from the waveforms without much human intervention.For 1000Hz audio data set the CNN  model showed a testing accuracy of 91.102\% for 0.2 second sample  window length while the LSTM model showed 93.999\% for the same. Since the input of the method is only the waveform, it is suitable for real-time processing, thus, the models can very well be used also as an onsite earthquake early warning system requiring a minimum amount of preparation time and workload.
![General Overview](https://github.com/Tonumoy/MFCCNet/blob/master/TimeSeriesVsSpectrogram%201.jpg)

## Data Description
Indian Earthquake acceleration data of the past 10 years and magnitudes approximately between 2-8 was collected from PESMOS (Program for Excellence in Strong Motion Studies, IIT Roorkee) and EGMB (Eastern Ghats Mobile Belt) Data of earthquakes with magnitudes ranging between 2-5 Collected at Geology and Geophysics Lab IIT Kharagpur, were also used. 
Before training the models, proper cleaning of the audio data was done and essential features from the data were then extracted using popular speech processing methodologies of Filter Banks and Mel-Frequency Cepstral Coefficients (MFCCs).



## Contributors
* Tonumoy Mukherjee tonumoymukherjee2@gmail.com

## License & Copyright
&#169; Tonumoy Mukherjee, Indian Insitute of Technology Kharagpur
> Licensed under the [MIT License](LICENSE).
