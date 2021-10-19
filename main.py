from prediction.predict import predict_with_random_user_input
# from training.train import train
from training.train import train
from validating.validation import validate


def main():
    train()
    # validate()
    # predict_with_random_user_input()


if __name__ == '__main__':
    main()


