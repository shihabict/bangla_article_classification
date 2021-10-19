import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

from common_function import remove_extra_space, remove_low_length_text, remove_non_bangle_word, get_tokenization, \
    label_encoding
from settings import MAX_INPUT_LENGTH, REPORT_PATH, MODEL_PATH, BASE_DIR, NUMBER_OF_SAMPLE


def cls_report(test_padded, model, testing_label_seq):
    predictions = model.predict(test_padded)
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(testing_label_seq, y_pred)

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index=['economy', 'education', 'entertainment', 'international', 'sports', 'state',
                                'technology'],
                         columns=['economy', 'education', 'entertainment', 'international', 'sports', 'state',
                                  'technology'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt='g')
    acc = accuracy_score(testing_label_seq, y_pred) * 100
    with open(f'{BASE_DIR}{REPORT_PATH}validation_report.txt', 'w') as file:
        file.write(f"VALIDATION ACCURACY: {acc}")
    print(f"VALIDATION ACCURACY: {acc}")
    plt.title('LSTM VALIDATION \nAccuracy: {0:.2f}'.format(acc))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f'{BASE_DIR}{REPORT_PATH}val_classification_report.png')
    plt.show()
    return y_pred


def create_actual_predicted_df(encoder, data, y_pred):
    y_pred = encoder.inverse_transform(y_pred)
    data['predicted_category'] = y_pred
    data = data.drop('text', axis=1)
    data.to_json(open(f"{BASE_DIR}DATA/actual_predicted.json", 'w'), orient='records', indent=4)
    # return data


def validate():
    with open(f'{BASE_DIR}{MODEL_PATH}tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    test_data = pd.read_csv(f"{BASE_DIR}DATA/BanglaMCT7/test.csv")
    test_data = test_data.sample(NUMBER_OF_SAMPLE)

    test_data = remove_extra_space(test_data, "cleanText")
    test_data = remove_low_length_text(test_data, "cleanText")
    # for i, row in test_data.iterrows():
    #     print(f"Text: {row['cleanText']}")
    #     print(f"Category: {row['category']}")
    #     print("__________________")
    #     if i == 10:
    #         break
    test_data = remove_non_bangle_word(test_data, 'cleanText')

    X = test_data['cleanText']
    y = test_data['category']

    model = load_model(f'{BASE_DIR}{MODEL_PATH}Model_lstm.h5')

    # Test tokenization
    _, test_sequences, test_padded = get_tokenization(tokenizer, X, MAX_INPUT_LENGTH)

    pkl_file = open(f'{BASE_DIR}{MODEL_PATH}encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    labels, class_names = label_encoding(le, test_data, 'category')
    testing_label_seq = labels
    y_pred = cls_report(test_padded, model, testing_label_seq)
    create_actual_predicted_df(le, test_data, y_pred)




if __name__ == '__main__':
    validate()
