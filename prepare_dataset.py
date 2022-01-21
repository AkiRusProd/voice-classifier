from tqdm import tqdm
from convert_audio2spect import load_audio,get_spectrogram,plot_spectrogram,resize_waveform
import pandas as pd




def prepare_data(df,data_path):

        for ind in tqdm(df.index,desc=data_path):

            gender = df['gender'].iloc[ind]
            
            waveform = load_audio('data/'+ df['filename'].iloc[ind])
            
            waveform = resize_waveform(waveform,waveform_length)
            spectrogram = get_spectrogram(waveform)
 
            plot_spectrogram(spectrogram.numpy(),data_path+gender+'/'+f'{gender}-{ind}')


waveform_length = 100000

train_df= pd.read_csv("train_dataframe.csv")
test_df= pd.read_csv("test_dataframe.csv")

prepare_data(train_df, 'prepared data/train/')
prepare_data(test_df, 'prepared data/test/')

