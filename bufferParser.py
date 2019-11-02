import re

def parse_file(file_path, min_value):
    buffer_file = open(file_path,'r')
    attribute_data = dict()
    file_line = buffer_file.readline()
    while file_line:
        raw_data = file_line.split()
        if float(raw_data[0]) > min_value:
            attribute = raw_data[2]
            try:
                word = re.search('([^_]*)$', attribute).group(1)
                attribute_data[word] = float(raw_data[0])
            except AttributeError:
                print("regex parsing error")
        file_line = buffer_file.readline()

    buffer_file.close()
    return attribute_data


def main():
    buffer_file = read_file('correlBuffer.txt')
    attribute_data = parse_file(buffer_file, 0)
    print(attribute_data)


if __name__ == "__main__":
    main()
