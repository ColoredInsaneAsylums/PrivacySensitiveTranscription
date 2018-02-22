# build a string dictionary to perform QbS/word recognition
def main():
    # path to Unix/Linux words file
    words_path = '/usr/share/dict/words'

    # IAM PHOC unigrams
    unigrams = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
                'w', 'x', 'y', 'z']

    # get words from Unix/Linux dictionary that do not contain extraneous unigrams
    with open(words_path, 'r') as handle:
        words = sorted(list(set([line.rstrip().lower() for line in handle \
                       if not set(list(line.rstrip().lower())) - set(unigrams)])))

    # save dictionary to file
    with open('words.txt', 'w') as handle:
        handle.write('\n'.join(words))

if __name__ == '__main__':
    main()

