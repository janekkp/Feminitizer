import numpy as np
from tensorflow import keras


def decode_sequence(input_seq, encoder, decoder, tokenizer_output_char_index, tokenizer_output_index_char):

    # encode the input sequence to get the internal state vectors.
    states_value = encoder.predict(input_seq)

    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, 35))
    target_seq[0, 0, tokenizer_output_char_index['\t']] = 1.

    # loop for producing feminative
    stop_condition = False
    predicted_feminative = ''
    while not stop_condition:
        output_tokens, h1, c1, h2, c2 = decoder.predict(
            [target_seq] + states_value)

        # add token to predicted word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer_output_index_char[sampled_token_index]
        predicted_feminative += sampled_char

        # if word is too long or next predicted character is "\n" stop predicting
        if sampled_char == '\n' or len(predicted_feminative) > 24:
            stop_condition = True

        # update target
        target_seq = np.zeros((1, 1, 35))
        target_seq[0, 0, sampled_token_index] = 1

        # update states
        states_value = [h1, c1, h2, c2]

    return predicted_feminative


def feminatize(words):

    tokenizer_output_char_index = {'\t': 3, '\n': 4, 'a': 1, 'b': 21, 'c': 13, 'd': 19, 'e': 11, 'f': 26, 'g': 22,
                                   'h': 24, 'i': 5, 'j': 23, 'k': 2, 'l': 14, 'm': 18, 'n': 6, 'o': 7, 'p': 17,
                                   'r': 8, 's': 12, 't': 9, 'u': 20, 'w': 15, 'y': 16, 'z': 10, 'ó': 28, 'ą': 31,
                                   'ć': 34, 'ę': 29, 'ł': 25, 'ń': 32, 'ś': 30, 'ź': 33, 'ż': 27}

    tokenizer_output_index_char = {1: 'a', 2: 'k', 3: '\t', 4: '\n', 5: 'i', 6: 'n', 7: 'o', 8: 'r', 9: 't',
                                   10: 'z', 11: 'e', 12: 's', 13: 'c', 14: 'l', 15: 'w', 16: 'y', 17: 'p', 18: 'm',
                                   19: 'd', 20: 'u', 21: 'b', 22: 'g', 23: 'j', 24: 'h', 25: 'ł', 26: 'f', 27: 'ż',
                                   28: 'ó', 29: 'ę', 30: 'ś', 31: 'ą', 32: 'ń', 33: 'ź', 34: 'ć'}

    tokenizer_input_char_index = {'a': 1, 'b': 19, 'c': 11, 'd': 17, 'e': 7, 'f': 24, 'g': 20, 'h': 22, 'i': 2,
                                  'j': 21, 'k': 10, 'l': 13, 'm': 16, 'n': 3, 'o': 4, 'p': 15, 'r': 5, 's': 9,
                                  't': 6, 'u': 18, 'w': 14, 'y': 12, 'z': 8, 'ó': 27, 'ą': 30, 'ć': 32, 'ę': 29,
                                  'ł': 23, 'ń': 25, 'ś': 28, 'ź': 31, 'ż': 26}

    encoder = keras.models.load_model('encoder_predict.h5')
    decoder = keras.models.load_model('decoder_predict.h5')

    out_string = ""
    words = words.split()
    user_array = np.zeros((len(words), 22, 33))
    for i, input_form in enumerate(user_array):
        for t, char in enumerate(words[i].lower()):
            user_array[i, t, tokenizer_input_char_index[char]] = 1.

    for seq_index in range(len(user_array)):
        input_seq = user_array[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder, decoder, tokenizer_output_char_index, tokenizer_output_index_char)

        out_string += decoded_sentence.strip()
        out_string += ", "

    out_string = out_string[:-2]
    return out_string