# voice-classifier
## Simply TF classifier model that recognize male/female speech.

### Scripts
`train_model.py` - model that trains to classify voice spectrograms
>
`prepare_dataframe.py` - create csv frames for model training and testing
>
`prepare_dataset.py` - convert mp3 audiofiles to spectrograms using `convert_audio2spect.py` script

`recognize_voice.py` - classify male/female speech from audio


This model was trained on a  [Mozilla Common Voice](https://www.kaggle.com/mozillaorg/common-voice) dataset. 
>Training data includes ~45000 files, testing data includes ~15000 files.
The accuracy is 92.5 percent on the testing data.
>
To convert mp3 audio files to wav and read them you need also to install [ffmpeg](https://www.ffmpeg.org/download.html).

