#from phocnet.attributes.phoc import build_phoc

def main():
    words_path = '/usr/share/dict/words'
    unigrams = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
                'w', 'x', 'y', 'z']

    with open(words_path, 'r') as handle:
        words = [line.rstrip().lower() for line in handle \
                 if not set(list(line.rstrip().lower())) - set(unigrams)]

    print(words)
    print(len(words))

if __name__ == '__main__':
    main()

