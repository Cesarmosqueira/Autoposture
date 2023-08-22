import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def initialize_model(model_path):
    loaded_model = load_model(model_path)
    return loaded_model

def preprocess_sequences(sequences):
    scaler = MinMaxScaler()
    normalized_sequences = np.zeros_like(sequences)
    print(sequences.shape)
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            # Flatten the landmarks for each set within the sequence
            landmarks_flattened = np.reshape(sequences[i, j], (-1, 1))
            # Normalize the landmarks
            landmarks_normalized = scaler.fit_transform(landmarks_flattened)
            # Reshape the normalized landmarks back to the original shape
            normalized_landmarks = np.reshape(landmarks_normalized, sequences[i, j].shape)
            # Update the normalized landmarks in the sequences array
            normalized_sequences[i, j] = normalized_landmarks
    return normalized_sequences

# if __name__ == '__main__':
#     model = initialize_model('src_models/lstm_model_v01.h5')

