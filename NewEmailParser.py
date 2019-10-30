# intended usage
# parsed_mails, counts = parse_emails(enron_mail)
# write_arff_file(parsed_mails, wordcount, "output.arff")

import Email
import Keywords

def parse_emails(csv_file_handle):
    """Parse passed csv into Email objects and a word count dict"""
    parsed_emails = list(map(lambda x: Email.Email(x), csv_file_handle))
    #if it won't work, try:
    # parsed_emails = list(map(lambda x: Email.Email(x), iter(csv_file_handle)))

    word_counts = __count_words(parsed_emails)
    return (parsed_emails, word_counts)

def write_arff_file(list_emails, word_counts, output_path):
    """Calculate the attribute values, generate a string representation and write as arff"""
    with open(output_path, "w+") as file:
        file.write(Keywords.SPAMBASE + Keywords.LINE_FEED)
         # TODO: implement attribute_headers function
        file.write(__attribute_headers(word_counts))
        file.write(Keywords.ATTRIBUTE_CLASS + Keywords.LINE_FEED)
        file.write(Keywords.DATA + Keywords.LINE_FEED)

        for email in list_emails:
            # attempt to rewrite get_data
            file.write(__stringify_email(email, word_counts))

def __count_words(list_emails):
    """Calculates the total count of each word across all emails"""
    word_counts = dict()
    for email in list_emails:
        for word in email.word_occurrences:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    return word_counts

def __attribute_headers(word_counts):
    """Returns a string representation for all attribute headers"""
    result = ""
    qm = r''' "''' # quotation marks

    line_template = "" + Keywords.ATTRIBUTE + qm + Keywords.WORD_FREQ + r"{}" + qm + Keywords.INT + Keywords.LINE_FEED
    for word in word_counts:
        if word == "\\":
            result += line_template.format(r"\\")
        else:
            result += line_template.format(word)

    for c in Keywords.CHARACTERS:
        result += Keywords.ATTRIBUTE + " \"" + Keywords.CHAR_FREQ + c + "\"" + Keywords.INT + Keywords.LINE_FEED

    result += Keywords.ATTRIBUTE + Keywords.EMAIL_LENGTH + Keywords.INT + Keywords.LINE_FEED
    
    return result

def __stringify_email(email, word_counts):
    """Returns a string representation of the email in the context of word_counts"""
    string = ""
    
    # each attribute needs to be written, write 0 if word doesn't appear
    for word in word_counts:
        if word in email.word_occurrences:
            string += str(email.word_occurrences[word]) + ","
        else:
            string += "0,"
    
    for c in Keywords.CHARACTERS:
        string += str(email.char_occurrences[c]) + ","

    string += str(email.email_length) + ","
    string += email.ham
    return string