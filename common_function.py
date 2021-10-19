import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from tqdm import keras

# keras.backend.clear_session()


# accuracy_threshold = 0.99
# vocab_size = 80000
# embedding_dim = 64
# max_length = 128
# num_category = 7


def plot_class_distribution(data, target):
    sns.set(font_scale=1.4)
    # plot news category
    data[target].value_counts().plot(kind='barh', figsize=(6, 4))
    plt.xlabel('Number of News', labelpad=12)
    plt.ylabel('Class', labelpad=12)
    plt.yticks(rotation=45)
    plt.title("Dataset Distribution", y=1.02)


def remove_extra_space(data, col_name):
    data[col_name] = data[col_name].apply(lambda x: re.sub(' +', ' ', x))
    return data


def remove_low_length_text(data, col_name):
    dataset = data.copy()
    # Length of each headlines
    dataset['length'] = dataset['cleanText'].apply(lambda x: len(x.split()))
    # Remove the headlines with least words
    dataset = dataset.loc[dataset.length > 10]
    dataset = dataset.reset_index(drop=True)
    print("After Cleaning:", "\nRemoved {} Small Headlines".format(len(data) - len(dataset)),
          "\nTotal Headlines:", len(dataset))
    return dataset


def remove_non_bangle_word(data, col_name):
    data[col_name] = data[col_name].apply(
        lambda x: "".join(i for i in x if i in ["।"] or 2432 <= ord(i) <= 2559 or ord(i) == 32))
    return data


def data_summary(dataset, col_name, target):
    documents = []
    words = []
    u_words = []
    total_u_words = [word.strip().lower() for t in list(dataset[col_name]) for word in t.strip().split()]
    class_label = [k for k, v in dataset[target].value_counts().to_dict().items()]
    # find word list
    for label in class_label:
        word_list = [word.strip().lower() for t in list(dataset[dataset[target] == label][col_name]) for word in
                     t.strip().split()]
        counts = dict()
        for word in word_list:
            counts[word] = counts.get(word, 0) + 1
        # sort the dictionary of word list
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        # Documents per class
        documents.append(len(list(dataset[dataset[target] == label][col_name])))
        # Total Word per class
        words.append(len(word_list))
        # Unique words per class
        u_words.append(len(np.unique(word_list)))

        print("\nClass Name : ", label)
        print("Number of Documents:{}".format(len(list(dataset[dataset[target] == label][col_name]))))
        print("Number of Words:{}".format(len(word_list)))
        print("Number of Unique Words:{}".format(len(np.unique(word_list))))
        print("Most Frequent Words:\n")
        for k, v in ordered[:10]:
            print("{}\t{}".format(k, v))
    print("Total Number of Unique Words:{}".format(len(np.unique(total_u_words))))

    return documents, words, u_words, class_label


def class_word_distribution(documents, words, u_words, class_names):
    data_matrix = pd.DataFrame({'Total Documents': documents,
                                'Total Words': words,
                                'Unique Words': u_words,
                                'Class Names': class_names})
    df = pd.melt(data_matrix, id_vars="Class Names", var_name="Category", value_name="Values")
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()

    sns.barplot(data=df, x='Class Names', y='Values', hue='Category')
    ax.set_xlabel('Class Names')
    ax.set_title('Data Statistics')

    ax.xaxis.set_ticklabels(class_names, rotation=45)


def check_news_text_length(dataset, col_name):
    # Calculate the Review of each of the Review
    dataset['textlineLength'] = dataset[col_name].apply(lambda x: len(x.split()))
    frequency = dict()
    for i in dataset.textlineLength:
        frequency[i] = frequency.get(i, 0) + 1

    plt.bar(frequency.keys(), frequency.values(), color="b")
    plt.xlim(1, 20)
    # in this notbook color is not working but it should work.
    plt.xlabel('Length of the Text')
    plt.ylabel('Frequency')
    plt.title('Length-Frequency Distribution')
    # plt.show()
    print(f"Maximum Length of a Text: {max(dataset.textlineLength)}")
    print(f"Minimum Length of a Text: {min(dataset.textlineLength)}")
    print(f"Average Length of a Text: {round(np.mean(dataset.textlineLength), 0)}")


def label_encoding(le, data, target_col):
    encoded_labels = le.transform(data[target_col])
    labels = np.array(encoded_labels)  # Converting into numpy array
    class_names = le.classes_  ## Define the class names again
    return labels, class_names


class color:  # Text style
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def dataset_split(data, labels):
    X = data['cleanText']
    y = labels
    # X, X_test, y, y_test = train_test_split(X_feature, y, train_size=0.9,
    #                                         test_size=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                          test_size=0.2, random_state=0)
    print(color.BOLD + "\nDataset Distribution:\n" + color.END)
    print("\tSet Name", "\t\tSize")
    print("\t========\t\t======")

    print("\tFull\t\t\t", len(X),
          "\n\tTraining\t\t", len(X_train),
          "\n\tTest\t\t\t", len(X_test))
          # "\n\tValidation\t\t", len(X_valid))

    return X_train,X_test, y_train, y_test


def get_tokenization(tokenizer, data, max_length):
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    # Train Data Tokenization
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(data)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
    return word_index, train_sequences, train_padded


# class myCallback(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#       if logs.get('accuracy')>accuracy_threshold:
#         print("\nReached %2.2f%% accuracy so we will stop trianing" % (accuracy_threshold*100))
#         self.model.stop_training = True
