import sys
from keras.models import load_model

sys.path.append('..')
sys.path.append('../landmark_extraction_yolo')


model = load_model('models/lstm_model_v01.h5')
print(model)
