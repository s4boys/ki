#import EmailParser.EmailParser as EmailParser
import EmailParser
import NewEmailParser as nep


def read_email(path):
    return open(path, "r")


def main():
    output_path = "spam.arff"
    enron_mail = read_email("short_enron4.csv")


    # parser = EmailParser.EmailParser()
    # parser.parse_emails(enron_mail)
    # #parser.filter_attributes_by_occurrence(5)
    # parser.write_arff_file(output_path)

    mails, wordcount = nep.parse_emails(enron_mail)
    nep.write_arff_file(mails, wordcount, output_path)


if __name__ == "__main__":
    main()
