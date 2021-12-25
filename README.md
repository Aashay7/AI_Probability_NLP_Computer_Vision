# CSCI-B551 FA2021 Assignment 3 Probability, NLP, and computer vision

## Group Members: Aashay Gondalia (aagond), Harsh Atha (hatha), Sai Hari Chandan Morapakala (saimorap)

## Part 1: Part-of-speech tagging

### 1.1 Problem Statement

A basic problems in Natural Language Processing is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). Sometimes this is easy: a sentence like “Blueberries are blue” clearly consists of a noun, verb, and adjective, since each of these words has only one possible part of speech (e.g., “blueberries” is a noun but can’t be a verb). But in general, one has to look at all the words in a sentence to figure out the part of speech of any individual word. For example, consider the — grammatically correct! — sentence: “Buffalo buffalo Buffalo buffalo
buffalo buffalo Buffalo buffalo.” To figure out what it means, we can parse its parts of speech:

![image](https://media.github.iu.edu/user/18130/files/6bbe4080-52ed-11ec-9003-174275417551)


### 1.2 Data

Each line consists of a sentence, and each word is followed by one of 12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark).

### 1.3 Approaches

We have 3 different approaches to solve the problem: Simplified (Naive Bayes), HMM (using Viterbi) and MCMC using Gibbs.

### 1.3a Simplifed (Using Naive Bayes)

A Simplified version where we estimate the highest probable tag for a particular word using Bayes Rule. It uses the below Bayes net and estimates th tag using given below formula.

![image](https://media.github.iu.edu/user/18130/files/f94d6080-52ec-11ec-87b6-6e15586fca14)


![image](https://media.github.iu.edu/user/18130/files/fd797e00-52ec-11ec-81f7-e5370d98b31d)


where si* is the most probable tag for given word Wi.

#### Training:
In this Section we count the occurrences of the word having a particular tag (bag_of_words in the code). We also count the number of times a particular tag has appeared in the training data (tag_cnt in code). Based on the count of word appearing totally and for the particular tag, we calculate the probability (Given by word_prob in the code) as well as the probability of the tag with respect to the whole corpus (tag_prob in code). 

We predict the tag as the one which has the highest probability for that word. This is a Naive approach and does not consider previous tags, words or position into consideration.

#### Results for Simplified Version

1. For bc.test: Word Accuracy: 93.92% Sentence Accuracy: 47.45%
2. For bc.test.tiny: Word Accuracy: 97.62% Sentence Accuracy:66.67%

This is a good Word accuracy but can still be improved by using Viterbi Algorithm in an HMM or using MCMC.

### 1.3b HMM Viterbi

In this set we try to calculate the Maximum A Posteriori(MAP) by incorporating dependencies between words. It uses the following Bayes Net and considers dependencies between 2 consecutive words.

![image](https://media.github.iu.edu/user/18130/files/08cca980-52ed-11ec-8701-12902c0e6aaa)


It finds the sequence using below formula

![image](https://media.github.iu.edu/user/18130/files/0f5b2100-52ed-11ec-893b-5c2ad1674a6f)


#### Training:

Along with already calculated emission probabilities, we also find the initial probability (Given by initial_prob) which is basically probability of that word starting the sentence. It also calculates the transition probability from 1 state to next state (Given by trans). Using these Values we do forward step calculation starting with initial for state s0 and then creating a Viterbi table (V_table) which has the sentence words as it's columns and all possible tags as it's rows. After each forward step calculation we pick the max value and based on that generate values for next step. 

As soon as we reach the last step, we have to do a backward propagation using the MAP value for last state and go backwards till we create the entire sequence, (given by viterbi_seq).

#### Results for HMM Viterbi Version

1.  bc.test: Word Accuracy: 94.86% Sentence Accuracy: 52.90%
2.  bc.test.tiny: Word Accuracy: 97.62% Sentence Accuracy:66.67%

This is the best approach among all 3 methods and gives the best accuracy for word and sentence.

### 1.3c MCMC using Gibbs Sampling

In this method we use the below Net. As this is not an HMM, we cannot use Viterbi and therefore we run Gibbs Sampling. The idea behind Gibbs Sampling is to run the same logic N times and create a new random sample every time. After running enough iterations, the sampling algorithm will converge and give us a good sample output. We have used below MCMC for this step.


![image](https://media.github.iu.edu/user/18130/files/171ac580-52ed-11ec-904e-4aee429d4e1d)



#### Training/Sampling

As seen in figure above we need to compute additional probabilities for this method. 1 state is connected to 2 future states as well as 2 states in the past (Given by trans2) and the current word is connected to the previous state (given by word_count_prev_prob). Ideal approach would be to calculate probabilities by just changing current state values while keeping all other states values constant. This is the original idea behind Gibbs, but calculating static values at each step for entire sentence, N times gets computationally heavy. So we do not calculate values not dependent on current state. In this case we have chosen a sampling value of 150. Usually we require thousands of iterations, but based on given datasets these converge for a value far smaller than 150. All samples for that particular sentence are stored in a list of lists. Using those values we find the tag occurring the most in it's particular position and append it to final prediction (given by predicted_list).


1. For bc.test: Word Accuracy: 93.92% Sentence Accuracy: 47.45%
2. For bc.test.tiny: Word Accuracy: 97.62% Sentence Accuracy:66.67%


### 1.4 Challenges/Assumptions

#### 1.4a Challenges:

a. In the 1st part of Bayes, there was not much of a challenge.

b. In Viterbi sequence, we tried to run the entire thing on our own and had some challenged in backward tracking. We referred to the sample code, given by Prof. Crandall in the in-class activity. Using the same flow of logic, we managed to succeed in getting a good output.

c. This was the most challenging part. Computing transition and emission for each state was an issue. Calculating probabilities for the chain for each position in the state was another challenge. Logic used had if-else condition on length of the sentence. In cases where length of sentence is 2/3, some conditions failed and gave errors. The logic had to be tweaked for this. Another problem was using np.random.choice. As the probabilities did not add up to 1, the formula failed. Had to look up this issue and normalize values (given in References). The last part of taking best value out of all samples for particular word was already done in B555 class and was easier to do.

#### 1.4b Assumptions:

a. Testing will only be done on Data for given 12 tags.

b. As samples are converging within 150 for given datasets, it is set to that value to ensure code does not time out. Factor can be changed as well.

#### 1.5 Results

![image](https://media.github.iu.edu/user/18130/files/1f730080-52ed-11ec-8a53-da35e9ccb93b)


Above results are as executed on Silo for bc.test. 
Run time: 6.5 mins.

![image](https://media.github.iu.edu/user/18130/files/d9b83700-52f0-11ec-9a48-368ebfe20732)


Above results are executed on Silo for bc.test.tiny
Run Time: 0.8s

#### 1.6 References:

1.	https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

2.	Viterbi Training part referred from In class activity solution provided by Prof. Crandall

3.	https://medium.com/data-science-in-your-pocket/pos-tagging-using-hidden-markov-models-hmm-viterbi-algorithm-in-nlp-mathematics-explained-d43ca89347c4

4.	https://newbedev.com/np-random-choice-probabilities-do-not-sum-to-1 For normalizing probabilities.

5.	https://en.wikipedia.org/wiki/Gibbs_sampling

6.	https://towardsdatascience.com/gibbs-sampling-8e4844560ae5

7.	Referred Code given by Prof Crandall in in class activity for Gibbs.

8.	Confirmed logic of not calculating probabilities of redundant nodes with Prof. Crandall after class. Ex: s3 is only connected to s1,s2,w2,w3,w4,s5,s6. Other than this,rest ignored. 

9.	Read this for understanding concepts: https://webfiles.uci.edu/msteyver/publications/SteyversGriffithsLSABookFormatted.pdf

10.	Referred to Slides in B555 Machine Learning course - Prof. Roni Khardon for Sampling and MCMC.

## Part 2: Ice tracking

### 2.1 Problem Statement

Detecting air-ice and ice-rock boundaries from the given radar echogram.

### 2.2 Approaches

We have 3 different approaches to solve the problem: Simplified Bayes Net, HMM and HMM with human feedback.

#### 2.2.1 Simplified Bayes Net

The structure of the bayes net used for this approach is shown below.

![simplified](/part2/DocumentationImages/simplified.JPG)

Here the observed variable is the column and states are the rows, but since there is no dependencies b/w states, state to state transition is not possible.
Therefore, for the air-ice boundary the emission probability given a column is 1 for the pixel carrying maximum normalized edge strength value in that column, for all the other pixels in the column the emission probability given a column is 1. Correspondingly the same logic is applied for the ice-rock boundary but with the constraint that the ice-rock boundary is always below air-ice boundary and the minimum distance between air-ice boundary and ice-rock boundary is 10 pixels.

#### 2.2.2 HMM

The structure of the bayes net used for this approach is shown below.

![HMM](/part2/DocumentationImages/hmm.JPG)

The observed variables and states are same as above. The transition probabilities for a given state is shown below.

![Transition_probabilities](/part2/DocumentationImages/transition_prob.JPG)

The assumption here is that given that a pixel exists on a boundary in a given column, the possible pixel in the next column is either in the same row as the previous or, a row above or below it, or two rows above or below it.

Since there is dependencies between states, the emission probability for the air-ice boundary is the scaled edge strength, here the scaling is done such that the all the emission probabilities in a given column sum to 1. Correspondingly the same logic is applied for the ice-rock boundary but with the constraint that the emission probabilities for the states from the top of the image till 9 pixels below the air-ice boundary is zero.

#### 2.2.3 HMM with human feedback

In this approach for each of the boundaries a point which lies on these boundaries is passed as input. While the previous approach is followed there are minor differences applied to it. First for the transition probability, if the next possible state is in the column in which the passed input exists then the only transition possible is from a state in previous column to the state (row) given in the input. Similarly the emission probability is equal to 1 for the input pixel passed and for all the other pixels in the same column the value is 0.

### 2.3 Challenges

Since the probabilities are very low values we faced the underflow problem, to overcome it the viterbi table is scaled after calculating the values of the all states for a given observation. The scalling is done such that the values of all states for a given observeration sum to one.

### 2.4 Results

Below are the few images containing the results.

For image 31.png with the human input for air_ice (23,22) and ice-rock (5,65).

Air-Ice boundary:

![31_air_ice_output](/part2/DocumentationImages/31_air_ice_output.png)

Ice-Rock boundary:

![31_ice_rock_output](/part2/DocumentationImages/31_ice_rock_output.png)

For image 16.png with the human input for air_ice (21,22) and ice-rock (99,41).

Air-Ice boundary:

![16_air_ice_output](/part2/DocumentationImages/16_air_ice_output.png)

Ice-Rock boundary:

![16_ice_rock_output](/part2/DocumentationImages/16_ice_rock_output.png)

### 2.5 References

1) https://en.wikipedia.org/wiki/Viterbi_algorithm

## Part 3: 


### 3.1 Problem Statement


Our goal is to extract text from a noisy scanned image of a document using the versatility of HMMs.
But the images are noisy, so any particular letter may be difficult to recognize. 
However, if we make the assumption that these images have English words and sentences, we can use statistical properties of the language to resolve ambiguities, just like in Part 2.


### 3.2 Data

The data used in this part are images with some text in them.
 1. The train image that contains letters. (No Noise.)('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' ')

![courier-train](https://media.github.iu.edu/user/18070/files/2f90dd00-52f9-11ec-9b37-477487d8b4d3)


 2. There are 20 test images with different text with added noise.
    'It is so ordered.'
  For Example : test_17_0.png

![test-17-0](https://media.github.iu.edu/user/18070/files/361f5480-52f9-11ec-8e7e-048d43d28d5f)


### 3.3 Approaches

This is an optical character recognition problem. We are using following algorithmic approaches to extract the text present in the image.
1. Simplified (Naive Bayes)
  
2. Hidden Markov Model (using Viterbi algorithm.)


### 3.3.1 Simplifed (Using Naive Bayes)


Using the simplified (Naive Bayes) version, the most probable image is estimated are head-on sub-image comparison. 


![image](https://media.github.iu.edu/user/18130/files/f94d6080-52ec-11ec-87b6-6e15586fca14)


![image](https://media.github.iu.edu/user/18130/files/fd797e00-52ec-11ec-81f7-e5370d98b31d)


The emission probability is estimated by comparing each test character pixel matrix with the pixel matrix of each train character and return the similarity that exists.
    
4 counts are maintained to extract the most information from the comparison.
1. matched_star   --------- count of matched '*' pixel in train and test sub-images.
2. matched_blank  --------- count of matched ' ' pixel in train and test sub-images.
3. matched_star_blank ----- count of instances where pixel in train is '*' and the same coordinate pixel is ' '
4. matched_blank_star ----- count of instances where pixel in train is ' ' and the same coordinate pixel is '*'

- matched_star_blank gives the idea about the missed information by the test image which can prove to be vital in prediction.
- matched_blank_star gives the idea about the noise in the image, where the pixel value in train is ' ' but for the same coordinate pixel the pixel value is '*', which should not be the case ideally (for same character in train sub-image and test sub-image). 

Using this, we can account for noise in the image as well, and hence the emission probabilities can be calculated for each pixel based on the summation of the exponentiation of the matching pixels over 1-noise and the exponentiation of the non-matching pixels over the noise. 

The noise estimation of the image was done using different functions referred from : https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image

The noise on average for the test images was approximately 11 %. The assumption taken here is that 10% of pixels in the test images are noisy. 
Additionally the weights for matched_star, matched_blank, matched_star_blank and matched_blank_star have been calibrated based on intuition and experimentation.
The emissions probability is calculated after applying Laplace Smoothing(alpha=1).

Weight assumption : 

```python
weights = {
    'matched_star' :        0.9,
    'matched_blank' :       0.32,
    'mismatch_star_blank' : 1.3,
    'mismatch_blank_star' : -0.5,
}
```

### 3.3.1.1 Emission Probabilites.

The heatmap gives us an idea about the emission probability of the character as explained in the above section. 

![emission](https://media.github.iu.edu/user/18070/files/38ce7980-52fa-11ec-855c-4130b448fed1)





### 3.3.2 Hidden Markov Model (using Viterbi Algorithm)

In the hidden markov model as given in the figure 1(b), the viterbi algorithm is used to estimate the most likely sequence with the help of transition probabilites and initial probabilities acquired from the train set. Using this information improves the model as prior information drives the prediction towards true value. 

![image](https://media.github.iu.edu/user/18130/files/08cca980-52ed-11ec-8701-12902c0e6aaa)


It finds the sequence using below formula

![image](https://media.github.iu.edu/user/18130/files/0f5b2100-52ed-11ec-893b-5c2ad1674a6f)


### 3.3.2.1 Initial Probabilities.

The graph gives us the idea about the frequency of the word being initialized with a given character. 

![init_prob](https://media.github.iu.edu/user/18070/files/b34ac980-52f9-11ec-8614-d1e61e7a0326)

### 3.3.2.1 Transition Probabilities.

The heatmap gives us an idea about the frequency of the character followed by another character. 

![image_3](https://media.github.iu.edu/user/18070/files/bb0a6e00-52f9-11ec-95d8-bcbc8570fbee)


#### Results for Simplified and HMM Viterbi.


<img width="1220" alt="Screenshot 2021-12-01 at 10 30 06 PM" src="https://media.github.iu.edu/user/18070/files/f73dce80-52f9-11ec-85e5-b264e33d1ebb">

For the test image 11, it can be seen that the simplified model predicts the image with a good confidence, however due to noise, it misidentifies some characters. 
The HMM model fixes this issue and upon usage of the initial and transition probabilities from 'bc.train', the characters are identified correctly.
<img width="1220" alt="Screenshot 2021-12-01 at 10 30 59 PM" src="https://media.github.iu.edu/user/18070/files/f442de00-52f9-11ec-8aee-28209aac1337">


### 3.4 Challenges/Assumptions

#### 3.4.1 Challenges:

1. The emission probabilities were calculated after consideration of 4 elements as highlighted in detail at (3.3.1). The weight was selected after experimentation. 
2. The noise assumption was tuned manually to output the best results.

#### 3.4.2 Assumptions:

1. It is assumed that the sub-image window is of the dimensions (25x14) pixels. Each train sub-image window is compared with test sub-image window to estimate the emission probabilities.

2. The train_set assumption is made wherein if the filename contains 'bc.train' or 'bc.test' then every second word in the word list will be skipped. 

#### 3.5 References:

1. http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Handouts/recitations/HMM-inference.pdf

2. https://www.pythonpool.com/viterbi-algorithm-python/

3. https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
 
