import logging
from os import getenv, path

from dotenv import load_dotenv
from flask import Flask
import requests

from src.controller import model_controller
import src.prediction_system as ps

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

app.register_blueprint(model_controller)

model_url = getenv('AUTOPOSTURE_MODEL_URL', '')
model_folder = getenv('MODELS_FOLDER', '')
model_name = getenv('MODEL_NAME', 'autoposture-model.h5')

def download_autoposture_model(url):
    model_name = path.basename(url)
    r = requests.get(url, allow_redirects=True)
    model_path = path.join(model_folder, model_name)
    open(model_path, 'wb').write(r.content)
    return model_path

if model_url != '':
    logger.debug(f'Model downlodaded at {download_autoposture_model(model_url)}')
else:
    if path.exists(path.join(model_folder, model_name)):
        logger.debug(f'{model_name} found at \'{model_folder}\'')
    else:
        logger.error(f'{path.join(model_folder, model_name)} not found')

ps.initialize_model()

