import Keywords
import re


class Email:

    def __init__(self,line):
        splitted_line = line.split(Keywords.DELIMITER_SC)
        self.email_length = 0
        self.word_occurrences = dict()
        self.char_occurences = dict()
        self.fill_occurrences(splitted_line[1:-1])
        self.ham = splitted_line[-1]
        #self.case_relation = 0.0


    # removes quotation marks and reduces series of spaces to a single space
    def clean_content(self,content):
        content = content.replace("\"","")
        content = re.sub("  +"," ",content)
        return content

    def concatenate_input(self,input):
        content = ""
        for part in input:
            content += part
        return content

    def fill_occurrences(self, input):
        content = self.concatenate_input(input)

        self.email_length = len(content)

        content = self.clean_content(content)

        self.count_special_characters(content)

        #self.count_case_relation(content)

        words = content.split(Keywords.DELIMITER_WS)
        for word in words:
            if (word != ""):
                if( word in self.word_occurrences):
                    self.word_occurrences[word] += 1
                else:
                    self.word_occurrences[word] = 1

    def count_words_in_emails(self):
        for word in self.words:
            if word in self.word_occurrences:
                self.word_occurrences[word] += 1
            else:
                self.word_occurrences[word] = 1

# useless, 99,9% of emails are lower case
#    def count_case_relation(self,content):
#        upper = sum(map(str.isupper, content))
#        lower = sum(map(str.islower, content))

#        content_length = upper + lower

#        self.case_relation = upper / content_length

    def count_special_characters(self,content):
        for c in Keywords.CHARACTERS:
            self.char_occurences[c] = content.count(c)

