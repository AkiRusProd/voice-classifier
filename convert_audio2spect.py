import os,subprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



def convert_to_wav(file_path):
    subprocess.call(['ffmpeg', '-i', file_path,
                   'temp.wav'],stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT)

def load_audio(file_path):
    try:
        os.remove('temp.wav')
    except: pass
    finally:
        convert_to_wav(file_path)

        file_path = 'temp.wav'
        
        audio_binary = tf.io.read_file(file_path)
        audio, sample_rate = tf.audio.decode_wav(audio_binary)
        waveform_tensor = tf.squeeze(audio, axis=None) #axis = -1

        waveform = waveform_tensor.numpy()

        return waveform

def wave_from_stereo2mono(waveform):
    return waveform.sum(axis=1) / 2

def resize_waveform(waveform, size):
    return np.resize(waveform,size)

def get_spectrogram(waveform):
    
    frame_length = 255
    frame_step = 128

    waveform = tf.cast(waveform, tf.float32)

    spectrogram = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)

    return spectrogram




def plot_spectrogram(spectrogram,path):
    fig, ax = plt.subplots(figsize=(1, 1),dpi = 130)

    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]

    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
    ax.axis('off')

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()