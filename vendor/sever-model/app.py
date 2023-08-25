from flask import Flask
from src.controller import model_controller

app = Flask(__name__)

app.register_blueprint(model_controller)


if __name__ == '__main__':
    app.run()
