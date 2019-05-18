import librosa
import os
import numpy as np

from typing import Dict, Tuple



def extract_audio_features(audio_path: str) -> Tuple[list,float]:
    data , sr = librosa.load(audio_path)
    onset_frames = librosa.onset.onset_detect(y=data, sr=sr)
    offset = librosa.frames_to_time(onset_frames, sr=sr)[-1]
    return librosa.feature.spectral_rolloff(data+0.01, sr=sr)[0], offset


def import_data(path: str) -> Dict[str,list]:

    data: dict = {} 
    data_count = 0
    max_len = 50

    for folder in os.listdir(path):
        if os.path.isdir(path+folder):
            data[folder] = []
            data_count += len(os.listdir(path+folder))
            for document in os.listdir(path+folder):
                audio_data, offset = extract_audio_features(path+folder+"/"+document)
                data[folder].append(audio_data[-max_len:])
    data["count&maxlen"] = [data_count,max_len]

    return data   


def create_training_dataset(data: Dict[str,list], label: int) -> Tuple[np.ndarray,np.ndarray]:
    data_count, max_len = data["count&maxlen"]
    training_data = np.zeros((data_count, max_len))
    acceleration_features = np.zeros((data_count,1),dtype = int)
    tram_type_features = np.zeros((data_count,1),dtype = int)
    index = 0
    for key_laber, key in enumerate(data):
        if key != "count&maxlen":
            for record in data[key]:
                training_data[index,:] = record
                acceleration_features[index] = label
                tram_type_features[index] =  key_laber
                index+=1

    return training_data, acceleration_features, tram_type_features        