import Keywords
import Email


class EmailParser:

    def __init__(self):
        self.all_words = dict()
        self.emails = list()

    def count_words_in_all_emails(self):
        for email in self.emails:
            for word in email.word_occurrences:
                if word in self.all_words:
                    self.all_words[word] += 1
                else:
                    self.all_words[word] = 1

    # changes order of dict, attention!
    def filter_attributes_by_occurrence(self,relevant_amount):
        for word in list(self.all_words):
            if (self.all_words[word] < relevant_amount):
                self.all_words.pop(word,None)

    def get_data(self,email):
        data_line = ""
        for word in email.word_occurrences:
            if word in self.all_words:
                self.all_words[word] = email.word_occurrences[word]

            
            #else:
                #print("Could not find " + word + " in all words")
        for word in self.all_words:
            data_line += str(self.all_words[word]) + ","
            self.all_words[word] = 0

        for c in Keywords.CHARACTERS:
            data_line += str(email.char_occurrences[c]) + ","

        data_line += str(email.email_length) + ","

        #data_line += str(email.case_relation) + ","

        data_line += email.ham
        return data_line

    def get_attributes(self):
        result =""
        for word in self.all_words:
            if (word == "\\"):
                result += Keywords.ATTRIBUTE + " \"" + Keywords.WORD_FREQ + "\\\\" + "\"" + Keywords.INT + Keywords.LINE_FEED
            else:
                result += Keywords.ATTRIBUTE + " \"" + Keywords.WORD_FREQ + word + "\"" + Keywords.INT + Keywords.LINE_FEED

        for c in Keywords.CHARACTERS:
            result += Keywords.ATTRIBUTE + " \"" + Keywords.CHAR_FREQ + c + "\"" + Keywords.INT + Keywords.LINE_FEED

        result += Keywords.ATTRIBUTE + Keywords.EMAIL_LENGTH + Keywords.INT + Keywords.LINE_FEED

        #result += Keywords.ATTRIBUTE + Keywords.CASE_RELATION + Keywords.REAL + Keywords.LINE_FEE

        return result

    def get_spambase_file(self,path):
        file = open(path,"w+")
        file.write(Keywords.SPAMBASE + Keywords.LINE_FEED)
        return file

    def set_occurrences_to_false(self,relative_occurrences):
        for key in relative_occurrences:
            relative_occurrences[key] = False
        return relative_occurrences

    def parse_emails(self,csv_file):
        self.emails = list()
        for line in csv_file:
            email = Email.Email(line)
            # trying to get empty emails in validation
            # if(not email.occurrences):
            #     print("email was empty, continuing")
            #     continue
            self.emails.append(email)
        self.count_words_in_all_emails()

    def write_arff_file(self,output_path):
        arff_file = self.get_spambase_file(output_path)

        arff_file.write(self.get_attributes())

        arff_file.write(Keywords.ATTRIBUTE_CLASS + Keywords.LINE_FEED)
        arff_file.write(Keywords.DATA + Keywords.LINE_FEED)

        for email in self.emails:
            arff_file.write(self.get_data(email))

        arff_file.close()









