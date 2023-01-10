import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio 
#Hyvä tapa sitoa Filepath jotta voidaan käyttää tulevaisuudes
emfaani_FILE = os.path.join('Data','emf','emfSound.wav')
eiaanet_FILE = os.path.join('Data','RandomSpookyVoice','hey.wav')

def load_wav_16k_mono(filename):
    """Elikkäs Jooh, ottaa audion ja converttaa sen 16kHz ja mono
    wav fileksi.Funktio odottaa filenamena "Pathiä" Wavefileen mikä
    määriteltiin ylempänä. """
    #Load encoded wav file
    file_contents = tf.io.read_file(filename)
    #Decode wav (tensors by channels)
    wav, sample_rate =tf.audio.decode_wav(file_contents,desired_channels=1)
    #removes trailing axis?
    wav = tf.squeeze(wav,axis=-1)
    sample_rate = tf.cast(sample_rate,dtype=tf.int64)
    #Goes from 44100hz to 16000hz- amplitude of the audio signal
    wav = tfio.audio.resample(wav,rate_in=sample_rate,rate_out=16000)
    return wav

wave = load_wav_16k_mono(emfaani_FILE)
nwave = load_wav_16k_mono(eiaanet_FILE)
#tekee file pathist
pos_FILET = os.path.join('Data','emf')
neg_FILET = os.path.join('Data','RandomSpookyVoice')

#Creates Tensorflow Datasets ja \*wav merkkaa sitä että hakee kaikki jotka loppuvat Wav fileen.

posdata = tf.data.Dataset.list_files(pos_FILET+'\*.wav')
negdata = tf.data.Dataset.list_files(neg_FILET+'\*.wav')
#Tekee positiivisista positiiviset ja negatiivisesta negatiiviset koulutus arvot ja tekee data paketin
positivies = tf.data.Dataset.zip((posdata,tf.data.Dataset.from_tensor_slices(tf.ones(len(posdata)))))
negativies = tf.data.Dataset.zip((posdata,tf.data.Dataset.from_tensor_slices(tf.ones(len(negdata)))))
data = positivies.concatenate(negativies)
#käy läpi itemit emf äänistä ja lisää listaan
lenghts = []
for file in os.listdir(os.path.join('Data','emf')):
    #tulee 16khz wave form
    tensor_wave = load_wav_16k_mono(os.path.join('Data','emf',file))
    lenghts.append(len(tensor_wave))





def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label



filepath, label = positivies.shuffle(buffer_size=10000).as_numpy_iterator().next()

spectrogram, label = preprocess(filepath, label)



data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(6)
data = data.prefetch(3)
train = data.take(5)
test = data.skip(2).take(2)



samples, labels = train.as_numpy_iterator().next()

samples.shape




