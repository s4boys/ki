# import EmailParser.EmailParser as EmailParser
import EmailParser
import NewEmailParser as nep
import bufferParser


def read_email(path):
    return open(path, "r")


def merge_dicts(a_dict, b_dict):
    for a in a_dict:
        b_dict[a] = 1
    return b_dict


def main():
    output_path = "spam.arff"
    enron_mail = read_email("enron.csv")

    # parser = EmailParser.EmailParser()
    # parser.parse_emails(enron_mail)
    # #parser.filter_attributes_by_occurrence(5)
    # parser.write_arff_file(output_path)

    mails, wordcount = nep.parse_emails(enron_mail)
    correlated_words = bufferParser.parse_file("correlBuffer.txt", 0.1)
    informative_words = bufferParser.parse_file("infoGain.txt", 0.1)

    relevant_words = merge_dicts(correlated_words, informative_words)

    nep.write_arff_file(mails, relevant_words, output_path)


if __name__ == "__main__":
    main()
