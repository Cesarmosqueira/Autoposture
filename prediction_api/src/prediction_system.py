import logging
from os import getenv, path

from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)

logger.info('Loading prediction system')
THRESHOLD = float(getenv('THRESHOLD', '0.7'))
models_folder = getenv('MODELS_FOLDER', '')
model_name = getenv('MODEL_NAME', 'autoposture-model.h5')
model = None

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

    prediction = model.predict(sequence, verbose=0).tolist()
    score = prediction[0][0]
    response = { 
                'score': score,
                'status': 'Good' if score > THRESHOLD else 'Bad'
    }
    return response

def initialize_model():
    global model
    model_path = path.join(models_folder, model_name)
    model = load_model(model_path)
    logger.info(model.summary())

def healthcheck():
    global model
    return True if model else False
