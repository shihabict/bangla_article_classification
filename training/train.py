import pickle
import seaborn as sns
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

from common_function import remove_extra_space, remove_low_length_text, remove_non_bangle_word, data_summary, \
    class_word_distribution, check_news_text_length, label_encoding, dataset_split, get_tokenization


from settings import NUMBER_OF_SAMPLE, MODEL_PATH, VOCAB_SIZE, EMBEDDING_DIMENSION, NUM_CALSSES, \
    NUM_EPOCHS, BATCH_SIZE, MAX_INPUT_LENGTH, REPORT_PATH, BASE_DIR


def create_model(vocab_size, embedding_dimension, input_length, num_category):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dimension, input_length=input_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(5),
        tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
        tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        keras.layers.Flatten(),
        tf.keras.layers.Dense(num_category, activation='softmax')])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def cls_report(test_padded, model, testing_label_seq):
    predictions = model.predict(test_padded)
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(testing_label_seq, y_pred)
    cs_report = classification_report(testing_label_seq, y_pred)

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index=['economy', 'education', 'entertainment', 'international', 'sports', 'state',
                                'technology'],
                         columns=['economy', 'education', 'entertainment', 'international', 'sports', 'state',
                                  'technology'])

    print(f"Confusion Matrix:{cm_df}")
    print(f"Classification Report:{cs_report}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt='g')
    plt.title('LSTM TRAINING \nAccuracy: {0:.2f}'.format(accuracy_score(testing_label_seq, y_pred) * 100))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f'{REPORT_PATH}classification_report.png')
    plt.show()


def save_accuracy(history):
    with open(f'{BASE_DIR}{REPORT_PATH}training_report.txt', 'w') as file:
        file.write(f"MODEL TRAINING ACCURACY: {mean(history.history['accuracy'])}")
        file.write('\n')
        file.write(f"MODEL TRAINING LOSS: {mean(history.history['loss'])}")
        print(f"Model accuracy and loss save to {REPORT_PATH}training_report.txt")


def train():
    train_data = pd.read_csv(f"{BASE_DIR}DATA/BanglaMCT7/train.csv")
    train_data = train_data.sample(NUMBER_OF_SAMPLE)
    # print(train_data['category'].value_counts())
    # label encoding
    le = LabelEncoder()
    le.fit(train_data['category'])

    #Save Label encoder
    output = open(f'{BASE_DIR}{MODEL_PATH}encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    print(f"Encoder save to {BASE_DIR}{MODEL_PATH}encoder.pkl")
    # preprocessing

    train_data = remove_extra_space(train_data, "cleanText")
    train_data = remove_low_length_text(train_data, "cleanText")
    train_data = remove_non_bangle_word(train_data, 'cleanText')
    # Data Visulization
    documents, words, u_words, class_names = data_summary(train_data, 'cleanText', 'category')
    class_word_distribution(documents, words, u_words, class_names)
    check_news_text_length(train_data, 'cleanText')

    # label encoding
    labels, class_names = label_encoding(le, train_data, 'category')

    # Split Dataset
    X_train, X_test, y_train, y_test = dataset_split(train_data, labels)

    # Tokenization
    # define tokenization
    oov_tok = "<OOV>"
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    # Save Token
    with open(f'{BASE_DIR}{MODEL_PATH}tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Token save to {BASE_DIR}{MODEL_PATH}tokenizer.pickle")
    # Load Token
    # with open(f'{BASE_DIR}{MODEL_PATH}tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)

    # Train toeknization
    _, train_sequences, train_padded = get_tokenization(tokenizer, X_train, MAX_INPUT_LENGTH)

    # Test tokenization
    _, test_sequences, test_padded = get_tokenization(tokenizer, X_test, MAX_INPUT_LENGTH)

    train_label_seq = y_train
    testing_label_seq = y_test

    # Saved the Best Model
    filepath = f"{BASE_DIR}{MODEL_PATH}Model_lstm.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                 save_weights_only=False, mode='max')
    callback_list = [checkpoint]

    model = create_model(VOCAB_SIZE, EMBEDDING_DIMENSION, MAX_INPUT_LENGTH, NUM_CALSSES)
    print(f"TRAINING STARTED")
    history = model.fit(train_padded, train_label_seq,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=callback_list)

    print(f"TRAINING ACCURACY: {np.mean(history.history['accuracy'])}")
    print(f"TRAINING LOSS: {np.mean(history.history['loss'])}")
    save_accuracy(history)
    # Save model
    # model.save(f'../MODELS/{filepath}')
    cls_report(test_padded, model, testing_label_seq)
    # save_model_history(history, MODEL_PATH)


if __name__ == "__main__":
    train()
