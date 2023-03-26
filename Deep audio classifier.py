#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 tensorflow-io matplotlib
get_ipython().system('pip install tensorflow-io')


# In[4]:


import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio


# In[5]:


arrest= os.path.join('C:/Users/Nandhini/Downloads/data/arrest/24161_TV.wav')
cardio= os.path.join('C:/Users/Nandhini/Downloads/data/cardio/13918_AV.wav')


# In[6]:


arrest


# In[7]:


#This function is used for loading and preprocessing WAV files for use in
#machine learning or other audio processing tasks.
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# In[8]:


wave = load_wav_16k_mono(arrest)
nwave = load_wav_16k_mono(cardio)


# In[9]:


plt.plot(wave)


# In[10]:


plt.plot(nwave)


# In[11]:


POS = os.path.join('cardio.wav')
NEG = os.path.join('arrest.wav')


# In[12]:


pos = tf.data.Dataset.list_files('cardio.wav')
neg = tf.data.Dataset.list_files('arrest.wav')


# In[13]:


# Print the first file in each dataset for verification
#verify that the dataset was created and that the first file path matches the expected path.
print('Positive file:', next(iter(pos)))
print('Negative file:', next(iter(neg)))


# In[14]:


# Add labels and combine positive and negative samples
positives = pos.map(lambda x: (x, 1))
negatives = neg.map(lambda x: (x, 0))
data = positives.concatenate(negatives)


# In[15]:


data


# In[23]:


#computing the lengths of all the WAV files in the directory and store them
lengths = []
for file in os.listdir(os.path.join('C:/Users/Nandhini/Downloads/data/arrest')):
    tensor_wave = load_wav_16k_mono(os.path.join('C:/Users/Nandhini/Downloads/data/arrest', file))
    lengths.append(len(tensor_wave))


# In[24]:


tf.math.reduce_mean(lengths)


# In[25]:


tf.math.reduce_min(lengths)


# In[26]:


tf.math.reduce_max(lengths)


# In[16]:


#this code is preprocessing the WAV files in a dataset for use in training a machine learning model
#padded WAV file to obtain a spectrogram, takes the absolute value of the spectrogram
#The resulting spectrogram and label are returned as outputs
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


# In[17]:


#code randomly selects a WAV file from the positives dataset, 
#preprocesses it using the preprocess function to obtain a spectrogram,
#and plots the spectrogram as an image
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()


# In[18]:


# The preprocess function extracts the spectrogram and applies some data augmentation
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)


# In[19]:


train = data.take(36)
test = data.skip(36).take(15)


# In[20]:


samples, labels = train.as_numpy_iterator().next()
samples.shape


# In[21]:


#Load Tensorflow Dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
#Build Sequential Model, Compile and View Summary
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.summary()


# In[27]:


hist = model.fit(train, epochs=2, validation_data=test)


# In[39]:


plt.title('Model fit')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['precision'], 'g')
plt.plot(hist.history['loss'], 'b')
plt.show()


# In[41]:


# loads an MP3 file, converts it to a float tensor, and resamples it to 16 kHz single-channel audio
def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav


# In[42]:


mp3 = os.path.join('cardio.wav')


# In[43]:


wav = load_mp3_16k_mono(mp3)


# In[44]:


#This code retrieves the first element
#This sequence will be used as an input to the machine learning model during training.
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)


# In[45]:


samples, index = audio_slices.as_numpy_iterator().next()


# In[46]:


#STFT-short time fourier transform
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


# In[47]:


audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)


# In[48]:


#The code is using a trained model to predict the output labels of the audio slices
yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]


# In[49]:


from itertools import groupby


# In[50]:


#calculates the number of times a positive class (1) is predicted by the model.
yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()
calls


# In[52]:


#model to predict the classes of audio samples from the "arrest" folder
results = {}
for file in os.listdir(os.path.join('C:/Users/Nandhini/Downloads/data/arrest')):
    FILEPATH = os.path.join('C:/Users/Nandhini/Downloads/data/arrest', file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat


# In[53]:


results


# In[54]:


class_preds = {} #binary classification predictions for each file
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits] #0.99-threshold value
#The resulting predictions are then stored as 1 (positive class)
#if the corresponding logit is greater than 0.99, and as 0 (negative class) 
class_preds


# In[55]:


postprocessed = {}   #contains the count of how many times the classifier detected a call
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed


# In[56]:


import csv


# In[57]:


with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['arrest'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])


# In[ ]:




