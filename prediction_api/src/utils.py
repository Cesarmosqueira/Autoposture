from numpy import array, zeros_like, reshape
from sklearn.preprocessing import MinMaxScaler

def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def preprocess_sequences(sequences):
    sequences = array(sequences)
    # Shape: (SeuenceSize, 51)
    scaler = MinMaxScaler()
    normalized_sequences = zeros_like(sequences)
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            # Flatten the landmarks for each set within the sequence
            landmarks_flattened = reshape(sequences[i, j], (-1, 1))
            # Normalize tshe landmarks
            landmarks_normalized = scaler.fit_transform(landmarks_flattened)
            # Reshape the normalized landmarks back to the original shape
            normalized_landmarks = reshape(landmarks_normalized, sequences[i, j].shape)
            # Update the normalized landmarks in the sequences array
            normalized_sequences[i, j] = normalized_landmarks
    return normalized_sequences
