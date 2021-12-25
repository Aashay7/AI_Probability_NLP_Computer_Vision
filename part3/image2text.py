#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Aashay Gondalia (aagond), Harsh Atha (hatha), Sai Hari (saimorap)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
#import matplotlib.pyplot as plt
#import scipy
#from scipy.signal import convolve2d

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    '''
    To load the image into a list of list of pixel matrix of each character in
    the image. 

    PARAMETERS : fname STRING
    RETURNS : result LIST
    '''
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result


def load_training_letters(fname):
    '''
    To load the train image in a dictionary.
    The key in this dictionary is the character from the possible (allowed) characters TRAIN_LETTERS
    The values in the dictionary is the list of list of image pixel matrix. 

    PARAMETERS : fname STRING
    RETURNS : DICTIONARY
    '''
    # Declaring TRAIN_LETTERS as a global variable to be accessed in other functions.
    global TRAIN_LETTERS
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


'''
# Function to estimate the noise in a image. Uses SCIPY. (Not used in this version.)
def estimate_noise(I):
    # Noise estimator from https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
    # Using the estimates noise from the function to calculate the emission probabilites results in poor predictions
    # for Naive Bayes and subseqently Hidden Markov Model using Viterbi. 
    H, W = I.shape
    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))   # convolve2d can be imported from scipy.
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
    return sigma
'''


def convert_to_num(train_char : list, test_char : list):
    '''
    Function to convert character to numpy array of 1's and 0's. (Used alongwith estimate_noise function)

    PARAMETERS : train_char LIST, test_char LIST
    RETURNS    : num_train_char np.array, num_test_char np.array 
    '''
    num_train_char = np.array([[1 if (train_char[x][y] == '*') else 0 for y in range(CHARACTER_WIDTH)] for x in range(CHARACTER_HEIGHT)])
    num_test_char = np.array([[1 if (test_char[x][y] == '*') else 0 for y in range(CHARACTER_WIDTH)] for x in range(CHARACTER_HEIGHT)])
    return num_train_char, num_test_char


def get_emission_prob(train_letters : list, test_letters : list) -> np.array:
    '''
    This is the function to calculate the emission probability. 
    The emission probability is estimated by comparing each test character pixel matrix
    with the pixel matrix of each train character and return the similarity that exists.
    
    4 counts are maintained to extract the most information from the comparison.
        1. matched_star   --------- count of matched '*' pixel in train and test sub-images.
        2. matched_blank  --------- count of matched ' ' pixel in train and test sub-images.
        3. matched_star_blank ----- count of instances where pixel in train is '*' and the same coordinate pixel is ' '
        4. matched_blank_star ----- count of instances where pixel in train is ' ' and the same coordinate pixel is '*'

    matched_star_blank gives the idea about the missed information by the test image which can prove to be vital in prediction.
    matched_blank_star gives the idea about the noise in the image, where the pixel value in train is ' ' but for the same
    coordinate pixel the pixel value is '*', which should not be the case ideally (for same character in train sub-image and 
    test sub-image). 
    
    FROM THE QUESTION FILE : 
    "
    If we assume that m% of the pixels are noisy, then a naive Bayes classifier could
    assume that each pixel of a given noisy image of a letter will match the corresponding pixel in the reference
    letter with probability (100 −m)%.
    "

    Using this, we can account for noise in the image as well, and hence the emission probabilities can be 
    calculated for each pixel based on the summation of the exponentiation of the matching pixels over 1-noise
    and the exponentiation of the non-matching pixels over the noise. The noise estimation of the image was 
    done using different functions referred from : https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image

    The noise on average for the test images was approximately 11 %. The assumption taken here is that 10% of pixels in the test images 
    are noisy. 
    Additionally the weights for matched_star, matched_blank, matched_star_blank and matched_blank_star have been calibrated 
    based on intuition and experimentation.

    The emissions probability is calculated after applying Laplace Smoothing(alpha=1).
    '''

    emissions = np.zeros((len(train_letters), len(test_letters)))
    total_pixels = CHARACTER_HEIGHT * CHARACTER_WIDTH
    
    '''
    # Best Weights for train_set.txt
    weights = {
        'matched_star' : 0.9,
        'matched_blank' : 0.3,
        'mismatch_star_blank' : 0.9,
        'mismatch_blank_star' : -0.5,
    }

    weights = {
        'matched_star' : 1.0,
        'matched_blank' : 0.26,
        'mismatch_star_blank' : 0.9,
        'mismatch_blank_star' : -0.5,
    }
    '''
    noise = 0.1
     
    weights = {
        'matched_star' :        0.9,
        'matched_blank' :       0.32,
        'mismatch_star_blank' : 1.3,
        'mismatch_blank_star' : -0.5,
    }
    
    for i in range(len(test_letters)):
        #test_image_stars = sum([sum([1 if (test_letters[i][x][y] == '*') else 0 for y in range(CHARACTER_WIDTH)]) for x in range(CHARACTER_HEIGHT)])
        #test_image_blanks = (CHARACTER_HEIGHT * CHARACTER_WIDTH) - test_image_stars
        #noise = estimate_noise(plt.imread(test_img_fname[i]))
        for j in range(len(train_letters)):  
            #sum_matched_pixels = sum([sum([1 if (test_letters[i][x][y] == list(train_letters.values())[j][x][y])  else 0 for y in range(CHARACTER_WIDTH)]) for x in range(CHARACTER_HEIGHT)])
            #sum_miss_pixels = total_pixels - sum_matched_pixels 
            matched_star = 0
            matched_blank = 0
            mismatch_star_blank = 0  # Train - Star ::: Test - Blank  ---- Missed important information
            mismatch_blank_star = 0  # Train - Blank ::: Test - Star  ---- Noise.
            train_letter = list(train_letters.values())[j]
            test_letter = test_letters[i]

            for m in range(CHARACTER_HEIGHT):
                for n in range(CHARACTER_WIDTH):
                    if train_letter[m][n] == test_letter[m][n]:
                        # Matched
                        if train_letter[m][n] == '*':
                            matched_star += 1
                        else:
                            matched_blank += 1
                    else:
                        # Mismatched
                        if train_letter[m][n] == '*' and test_letter[m][n] == ' ':
                            # Train : '*', Test : ' ', Information missed by the test sub-image. 
                            mismatch_star_blank += 1
                        else:
                            # Train : ' ', Test : '*', Information that represent the noise in the test sub-image. 
                            mismatch_blank_star += 1

            emissions[j][i] = (1-noise) * (matched_star * weights['matched_star'] \
                +  matched_blank * weights['matched_blank'] )\
                +  noise * (mismatch_star_blank * weights['mismatch_star_blank'] \
                +  mismatch_blank_star * weights['mismatch_blank_star']) 

    emissions = (emissions + 1) / (total_pixels + len(TRAIN_LETTERS)) 
    return emissions


def preprocess_text(train_set_filename : str) -> list:
    '''
    Create a word list from the train text file. 
    The function generates the word list based on the input file name.
    If the filename has 'bc.train' or 'bc.test' as a substring in the filename,
    every second word will be skipped. Adapted 
    
    PARAMETERS : train_set_filename STRING
    RETURNS : words LIST
    '''
    words = []
    if (('bc.train' in train_set_filename.lower()) or ('bc.test' in train_set_filename.lower())): 
        # For train-set files from part1. 
        # 'bc.train', 'bc.test', 'bc.test.tiny'
        # Refactored code snippet from A3P1 Skeleton code. 
        with open(train_set_filename, 'r') as file : 
            for line in file:
                data =[w for w in line.split()]
                words += data[0::2]
                words += ' '
        file.close()
    else:
        with open(train_set_filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                words += [word + ' ' for word in line.split()]
                #words += [word for word in line.split()]
        file.close()
    return words           
    

def get_initial_probs(words : list):
    '''
    Function to get initial probabilities of each character in the TRAIN_LETTERS. 
    
    The initial probabilities are computed from the train set. The initial probabilities 
    of a given character is typically the occurence of word starting with that character. 

    Laplace Smoothing is applied on the initial probabilities to calculate the initial probabilities.

    PARAMETERS  : words LIST
    RETURNS     : intial_probs np.array

    '''
    initial_probs_raw = [0] * len(TRAIN_LETTERS)

    for word in words:
        if word[0] in TRAIN_LETTERS:
            initial_probs_raw[TRAIN_LETTERS.index(word[0])] += 1
    #dict_a = {TRAIN_LETTERS[i] : initial_probs_raw[i] for i in range(len(TRAIN_LETTERS))}
    
    # Laplace(alpha= +1) Smoothing.
    initial_probs = np.array([((initial_probs_raw[i_p] + 1) / (len(words) + len(TRAIN_LETTERS))) for i_p in range(len(initial_probs_raw))])
    return initial_probs


def get_transition_probs(words):
    '''
    The transition probabilities are computed in this function from the character list. 
    A numpy matrix (72 x 72) handles the transition probabilites for going from a character in TRAIN_LETTERS 
    to another character in TRAIN_LETTERS. 

    PARAMETERS  : words LIST
    RETURNS     : transition_probs np.array
    '''
    chars = [char for word in words for char in word]   # Character list to estimate transition probabilities.
    #    total occurence of a char 'x' given another char 'y' has occured just before it.
    #    BAYES LAW : P(x | y) = P(y | x) . P(x)  /  P(y)
    transition_probs_raw = np.zeros((len(TRAIN_LETTERS), len(TRAIN_LETTERS)))
    for i in range(len(chars)-1):
        if chars[i] in TRAIN_LETTERS and chars[i+1] in TRAIN_LETTERS:
            transition_probs_raw[TRAIN_LETTERS.index(chars[i]), TRAIN_LETTERS.index(chars[i+1])] += 1
    transition_denom = np.sum(transition_probs_raw, axis=1)
    transition_probs =  np.array([[(transition_probs_raw[i,j] + 1) / (transition_denom[i] + len(TRAIN_LETTERS))  for j in range(len(TRAIN_LETTERS))] for i in range(len(TRAIN_LETTERS))])
    return transition_probs
    

def simple_bayes_net(test_letters : list, emission_prob : np.array) -> list:
    '''
    The simple bayes net as shown in the figure 1(a.) uses the emission probabilities without 
    prior knowledge of previous character to figure out the image. 
    The max element in each column from emission probability matrix is essentially the most likely 
    character based on naive bayes assumption. 

    PARAMETERS  : test_letters LIST, emission_prob np.array
    RETURNS     : char_preds LIST

    '''
    preds = np.argmax(emission_prob, axis=0)
    char_preds = [TRAIN_LETTERS[p] for p in preds]
    return char_preds


def hidden_markov_model_viterbi(train_letters, test_letters, initial_prob, transition_prob, emission_prob):
    '''
    The function implements the Viterbi algorithm for Hidden markov chain as shown in the figure 1(b.) to find
    the Maximum a posteriori (MAP) inference. 
    Each prediction is dependent on the prior information and prior prediction. 
    
    
    REFERENCE : This code is numpy refactored version of the Viterbi code shared by Prof. Crandall on Canvas.

    The algorithm uses logarithmic addition instead of real multiplication to calculate the probabilites. 
    The weight has been applied to tune the influence of initial and transition probabilities in the prediction
    of the character. 

    ADDITIONAL REFERENCE FOR INTUITION : https://www.pythonpool.com/viterbi-algorithm-python/

    PARAMETERS : train_letters LIST, test_letters LIST, initial_prob np.array (1x72), 
               : transition_prob np.array (72 x test_image_char_length), emission_prob np.array (72x72)
    '''
    
    # Get the length of the observation data and length of the training images to create a 2D arr for viterbi calculations
    # Initalization of Viterbi tables and the table to log the best possible path.
    viterbi_table = np.zeros((len(train_letters), len(test_letters)))
    best_path = np.zeros_like(viterbi_table, dtype=np.int16)

    weight = 0.0018
    # Initialization of viterbi table using initial probabilities and the transition probability for the first char/letter/sub-image.
    for i in range(len(train_letters)):
        viterbi_table[i,0] =  weight * np.log(initial_prob[i]) + np.log(emission_prob[i,0])
        #viterbi_table[i,0] = initial_prob[i] * emission_prob[i,0]
    
    # Forward pass : to calculate probabilities. 
    # Using transition probabilities and emission probabilities for the subsequent chars/letters.
    for i in range(1,len(test_letters)):
        for j in range(len(train_letters)):
            temp = [ (viterbi_table[t,i-1] + (weight * np.log(transition_prob[t, j])) + ( np.log(emission_prob[j,i]))) for t in range(len(train_letters)) ]
            #temp = [ (viterbi_table[t,i-1] * transition_prob[t, j] * emission_prob[j,i]) for t in range(len(train_letters)) ]
            best_s = max(temp)
            viterbi_table[j,i] = best_s
            best_path[j,i] = temp.index(best_s)
        
    # Reverse pass : to find the best possible sequence. Stored in best_sequence_num list.
    best_sequence_num = np.zeros(len(test_letters), dtype=np.int16)
    best_sequence_num[-1] = np.argmax(viterbi_table[:,-1])
    for i in np.arange(len(test_letters)-1, 0, -1): 
        best_sequence_num[i-1] = best_path[best_sequence_num[i],i]
    
    
    best_sequence = [TRAIN_LETTERS[best_state] for best_state in best_sequence_num]
    return best_sequence


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)


word_list = preprocess_text(train_txt_fname)

emission_probs = get_emission_prob(train_letters, test_letters)

# 1. Simple Bayes net ----------------------------------- Fig. 1(b)
simple_bayes_net_op = simple_bayes_net(test_letters, emission_probs)


# 2. Hidden Markov Model with MAP inference (Viterbi) --- Fig. 1(a)
# Calculate Initial probabilities from the word list
initial_probs = get_initial_probs(word_list)

# Calculate transition probabilities ...
transition_probs = get_transition_probs(word_list)

# Calculate the Viterbi table and estimate the best(most probable) path/sequence.
hmm_op = hidden_markov_model_viterbi(train_letters, test_letters, initial_probs, transition_probs, emission_probs) 

print("Simple: " + "".join(simple_bayes_net_op))
print("   HMM: " + "".join(hmm_op)) 



