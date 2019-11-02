import EmailParser
import NewEmailParser as nep
import Email
from sklearn.naive_bayes import GaussianNB
# from sklearn import preprocessing

def main():
    # parse
    with open("short_enron4.csv") as enron_mail:
        mails, wordcount = nep.parse_emails(enron_mail)


    # generate attributes
    labels = list(map(lambda x: x.ham, mails))
    # features = list(map(lambda x: (x.email_length, 1), mails))
    features = list(map(lambda mail: Email.AttributeSet(mail, wordcount).as_list, mails))
    
    # calculate split
    split = 0.99
    split_index = int(len(features) * split)

    # train
    model = GaussianNB()
    model.fit(features[:split_index], labels)

    # test / predict
    predicted = model.predict(features[split_index:][0])
    print("Predicted: ", predicted, " for ", predict_sample)

if __name__ == "__main__":
    main()
