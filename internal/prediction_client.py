import requests

HOST = 'localhost'
PORT = '8103'
def predict_http_request(payload):
    """
    Args:
        - payload: {'array': (1, 10, 50) shape (10 frames)}
    Returns:
        - score: Value between 0 and 1
        - status: Good or bad posture (depending on threshold:0.7)
    """
    response = requests.post(f"http://{HOST}:{PORT}/predict", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        print(response.text)
