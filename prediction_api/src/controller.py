from flask import Blueprint, request, jsonify
from src.prediction_system import get_prediction, healthcheck
import json

model_controller = Blueprint('controller', __name__)

@model_controller.route('/predict', methods=['POST'])
def prediction_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Invalid JSON data'}), 400

    data = json.loads(request.data)

    if 'array'in data:
        response = get_prediction(data)
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid or missing "array" field'}), 400


@model_controller.route('/healthcheck', methods=['GET'])
def healthcheck_endpoint():
    return jsonify({'model': healthcheck()}), 400
    

