import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

THRESHOLD = 0.6
model_path = '../src_models/lstm_model_ex1.h5'
model = load_model(model_path)

def model_ready():
    if not model:
        print('Couldn\'t load model')
        return model
    else:
        print("Model summary")
        print(model.summary())

def preprocess_sequences(sequences):
    sequences = np.array(sequences)
    scaler = MinMaxScaler()
    normalized_sequences = np.zeros_like(sequences)
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            # Flatten, Normalize, Reshape
            landmarks_flattened = np.reshape(sequences[i, j], (-1, 1))
            landmarks_normalized = scaler.fit_transform(landmarks_flattened)
            normalized_landmarks = np.reshape(landmarks_normalized, sequences[i, j].shape)
            # Save process in empty ndarray
            normalized_sequences[i, j] = normalized_landmarks

    return normalized_sequences


def get_prediction(payload : dict):
    received_array = np.array(payload['array'])
    sequence = preprocess_sequences(received_array)

    prediction = model.predict(sequence).tolist()
    score = prediction[0][0]
    response = { 
                'score': score,
                'status': 'Good' if score > THRESHOLD else 'Bad'
    }
    return response
