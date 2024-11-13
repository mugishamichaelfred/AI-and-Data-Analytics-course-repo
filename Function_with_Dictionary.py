def word_indices(sentence):
    words = sentence.split()
    index_dict = {}  # To store word indices

    # Loop through each word and its index
    for index, word in enumerate(words):
        # If the word is already in the dictionary, append the index
        if word in index_dict:
            index_dict[word].append(index)
        # If the word is not in the dictionary, add it with the current index in a list
        else:
            index_dict[word] = [index]
    
    return index_dict

sentence = "the AI and Data Analytics certification course that I am taking and at the end I will get certification"
print(word_indices(sentence))
