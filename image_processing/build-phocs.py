import _pickle as pickle

from phocnet.attributes.phoc import build_phoc

# build a PHOC dictionary to perform QbS/word recognition
def main():
    # IAM PHOC unigrams
    unigrams = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
                'w', 'x', 'y', 'z']

    # get words from dictionary
    with open('./words.txt', 'r') as handle:
        words = list(handle)

    # build PHOCs for dictionary using IAM PHOC parameters
    phocs = build_phoc(words=words,
                       phoc_unigrams=unigrams,
                       unigram_levels=[1, 2, 3, 4, 5])

    # save dictionary to file
    with open('./dictionary.pickle', 'wb') as handle:
        pickle.dump(dict(zip(words, phocs)), handle)

if __name__ == '__main__':
    main()

