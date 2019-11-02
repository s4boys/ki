import EmailParser
import NewEmailParser as nep
import Email
from sklearn.naive_bayes import GaussianNB
# from sklearn import preprocessing

def main():
    # parse
    with open("short_enron4.csv") as enron_mail:
        mails, wordcount = nep.parse_emails(enron_mail)

    # TODO: generate attributes that are currently
    # only being generated when writing the arff file

    # train
    labels = list(map(lambda x: x.ham, mails))
    # TODO: pass more attributes (need to be generated first)
    features = list(map(lambda x: (x.email_length, 1), mails))
    model = GaussianNB()
    model.fit(features, labels)

    # predict
    predict_sample = [3]
    predicted = model.predict([predict_sample])
    print("Predicted: ", predicted, " for ", predict_sample)


if __name__ == "__main__":
    main()
