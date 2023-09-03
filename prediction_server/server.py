import asyncio
import json

import websockets

from predictions import get_prediction, model_ready

PORT = 8765
HOST = '127.0.0.1'


async def handle_prediction_request(websocket):
    raw_response = await websocket.recv()
    try:
        payload_in = json.loads(raw_response)
    except json.JSONDecodeError:
        print("Couldn't unmarshall json")

    prediction_response = get_prediction(payload_in)

    if websocket.open:
        try:
            await websocket.send(json.dumps(prediction_response))
            print("Response sent:", prediction_response)
        except Exception as e:
            print("Error sending response:", e)

        else:
            print("WebSocket connection is closed.")

async def main():
    server = await websockets.serve(handle_prediction_request, HOST, PORT)
    await server.wait_closed()

if __name__ == "__main__":
    if model_ready():
        print('Model is ready')
    else:
        print('Check model path')
    asyncio.run(main())

