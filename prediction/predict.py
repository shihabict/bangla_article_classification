import pickle
from tensorflow.keras.models import load_model
from settings import MODEL_PATH, MAX_INPUT_LENGTH, BASE_DIR
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np


def predict_with_random_user_input():
    with open(f'{BASE_DIR}{MODEL_PATH}tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = load_model(f'{BASE_DIR}{MODEL_PATH}Model_lstm.h5')
    pkl_file = open(f'{BASE_DIR}{MODEL_PATH}encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    input_sentence = input('Enter input news: ')

    Xi_token = tokenizer.texts_to_sequences([input_sentence])
    Xi_pad = pad_sequences(Xi_token, padding='post', maxlen=MAX_INPUT_LENGTH)
    print('Model prediction')
    preds = model.predict(Xi_pad)
    # print('Confidence :')
    # print(preds)
    preds = preds
    total = 0
    # for k in range(len(preds[0])):
    #     print(le.inverse_transform([[k]]))
    #     print('%f %%' % (preds[0, k] * 100))
    #     total += preds[0, k] * 100
    # print(total)
    # print('Predicted class: %s'%(encoder.inverse_transform(model.predict_classes(Xi_pad))))
    predicted_cat = le.inverse_transform(np.argmax(preds, axis=1))
    print(f'Predicted class: {predicted_cat}')


