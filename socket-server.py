import json
import asyncio
import websockets
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model_path = 'pose_estimation_models/lstm_model_ex1.h5'
model = load_model(model_path)

print(model.summary())

def preprocess_sequences(sequences):
    sequences = np.array(sequences)
    # Shape: (SeuenceSize, 51)
    scaler = MinMaxScaler()
    normalized_sequences = np.zeros_like(sequences)
    print(sequences.shape)
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            # Flatten the landmarks for each set within the sequence
            landmarks_flattened = np.reshape(sequences[i, j], (-1, 1))
            # Normalize tshe landmarks
            landmarks_normalized = scaler.fit_transform(landmarks_flattened)
            # Reshape the normalized landmarks back to the original shape
            normalized_landmarks = np.reshape(landmarks_normalized, sequences[i, j].shape)
            # Update the normalized landmarks in the sequences array
            normalized_sequences[i, j] = normalized_landmarks
    return normalized_sequences


async def hello(websocket):


    payload_json = await websocket.recv()
    payload = json.loads(payload_json)

    received_array = np.array(payload['array'])
    processed_array = preprocess_sequences(received_array)

    single_sequence = np.expand_dims(processed_array, axis=0)

    sequence_prediction = model.predict(single_sequence)
    response = {'response_array': sequence_prediction.tolist()}
    
    if websocket.open:
        try:
            await websocket.send(json.dumps(response))
            print("Response sent:", response)
        except Exception as e:
            print("Error sending response:", e)

        else:
            print("WebSocket connection is closed.")

    print(sequence_prediction)

    # await websocket.send(response)
    print(f'Server sent: {response}')

async def main():
    async with websockets.serve(hello, "localhost", 8765):
        await asyncio.Future() # run forever

if __name__ == "__main__":
    asyncio.run(main())

