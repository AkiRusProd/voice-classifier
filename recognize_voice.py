from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

from convert_audio2spect import load_audio,get_spectrogram,plot_spectrogram,resize_waveform



def load_spectrogram(filename):
    
    img = load_img(filename, target_size=(image_size, image_size))
    img = img_to_array(img)

    img = img.reshape(1, image_size, image_size, 3)
    img = img/255
    img = img.astype('float32')

    return img
 

def recognize_voice(path):

    waveform = load_audio(path)
    
    waveform = resize_waveform(waveform,waveform_length)
    spectrogram = get_spectrogram(waveform)

    plot_spectrogram(spectrogram.numpy(),'temp_spect.png')

    img = load_spectrogram('temp_spect.png')

    result = model.predict(img)

    print('Male voice ' if result[0] > 0.5 else 'Female voice ')
    print(f'output neuron value: {result[0]} (>0.5 - male; <0.5 - female)')
 


image_size = 100
waveform_length = 100000

model = load_model('voice_recognition_model')


recognize_voice('calm.mp3')
recognize_voice('kid.mp3')
recognize_voice('out.wav')
recognize_voice('ytmale.mp3')
