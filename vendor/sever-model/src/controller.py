from flask import Blueprint


model_controller = Blueprint('controller', __name__)

@model_controller.route('/hello')
def hello():
    return "Homla"
