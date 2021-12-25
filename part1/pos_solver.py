###################################
# CS B551 Fall 2021, Assignment #3
#
# Your names and user ids: Harsh Atha(hatha), Aashay Gondalia (aagond), Sai Hari Morap (saimorap)
#
# (Based on skeleton code by D. Crandall)
#


## References
#https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
# Viterbi Training part referred from In class activity solution provided by Prof. Crandall
#https://medium.com/data-science-in-your-pocket/pos-tagging-using-hidden-markov-models-hmm-viterbi-algorithm-in-nlp-mathematics-explained-d43ca89347c4
# https://newbedev.com/np-random-choice-probabilities-do-not-sum-to-1 For normalizing probabilities.
# https://en.wikipedia.org/wiki/Gibbs_sampling
# https://towardsdatascience.com/gibbs-sampling-8e4844560ae5
# Referred Code given by Prof Crandall in in class activity for Gibbs.
# Confirmed logic of not calculating probabilities of redundant nodes with Prof. Crandall after class. Ex: s3 is only connected to s1,s2,w2,w3,w4,s5,s6. Other than this,rest ignored. 
# Read this for understanding concepts: https://webfiles.uci.edu/msteyver/publications/SteyversGriffithsLSABookFormatted.pdf
# Referred to Slides in B555 Machine Learning course - Prof. Roni Khardon for Sampling and MCMC.


import random
import math
import numpy as np
# TODO : Changes
from copy import deepcopy

class Solver:
    
    def __init__(self):
        self.bag_of_words={}         # Contains all words:{tags:count}
        self.tag_cnt={}              # Contains count of all tags {tag:count}
        self.tag_prob={}             # Contains probabilities of all tags based on counts above
        self.word_prob={}            # Contains emission probabilities words:{tags:prob} where prob = count of word for tag/total count of that tag
        self.word_count_prevtag={}   # Contains emission counts for MCMC, where 1 word is dependent on current and prev state
        self.trans_prev_cur={}       # Contains counts of current and prev tag combination
        self.initial ={}             # Contains count of tag which start a sentence
        self.initial_prob={}         # Contains probabilities of tag starting a sentence. Based on above dictionary counts
        self.trans={}                # Transition probability. Probability of going from state 1 to state2
        self.trans2={}               # Contains 3 levelled Transitions probability. Probability of s3 depending on s2 and s1
        self.word_count_prev_prob={} # Contains emission probabilities for MCMC where the word is dependent on current and previous state values.
        self.tag_combo_count={}      # Count of consecutive tag combos. Count of i-2,i-1 combo.
    
    
# ****************************************************************************************************************************************************************

# Below function is used in training for computing count of all Transitions, words, emissions. 
# Most count dictionaries are kept unchanged in case they need to be reused in other functions.
# Counting done for following scenarios:
    # 1. Count of initial prob. If a tag starts a sentence, its count is updated. Given by function initial_calc.
    # 2. Word Count. Count of the tag for a particular tag. Ex: Word name occurs 10 times as noun and 3 times as verb. {'name':{'noun':10, 'verb':3}}. Given by function count_words.
    # 3. Count of tags. Gives total count of a particular tag. Ex: {'noun':100}. Given by count_tags
    # 4a. Transition Calculation from 1 state to another. Given by function calc_trans. Calculates counts of going from s1,s2 and so on.....
    # 4b. Multiple State transition calculates count from 1 state to combination of 2 previous states. It is dependent on 3 states rather than 1 like previous case. Given by calc_trans2
    # 5. Count of current word with current and prev tag combination. Ex: s1= 'noun' w2='the' s2= 'adp'. It calculates count of occurrences where the is adp and previous state is noun. Given by word_count_tag_prev.

# ****************************************************************************************************************************************************************
    
    def counting(self, data):

        for i in range(len(data)):
            word_list, tag_list  = data[i]
            
            for j in range(len(word_list)):                
                curr_word = word_list[j]
                curr_tag = tag_list[j]

                if j<len(tag_list)-1:
                    next_tag=tag_list[j+1]
                
                if j>1:
                    tag_combo=tag_list[j-1]+"<>"+tag_list[j-2]
                    self.calc_trans2(curr_tag,tag_combo)
                
                if j!=0:
                    prev_tag=tag_list[j-1]
                    self.word_count_tag_prev(curr_word,curr_tag,prev_tag)
                    
                # self.initial probability.
                if j == 0:
                    self.initial_calc(curr_tag)

                self.count_words(curr_word, curr_tag)
                
                self.count_tags(curr_tag)
                
                self.calc_trans(curr_tag,next_tag)
                
    def initial_calc(self,tag):
        if tag in self.initial:
            self.initial[tag]+=1
        else:
            self.initial[tag]=1

            
    def count_words(self, word, tag):
        if word in self.bag_of_words:
            if tag in self.bag_of_words[word]:
                self.bag_of_words[word][tag]+=1
                self.word_prob[word][tag]+=1
            else:
                self.bag_of_words[word][tag]=1
                self.word_prob[word][tag]=1
        else:
            self.bag_of_words[word]={tag:1}
            self.word_prob[word]={tag:1}


    def count_tags(self,tag):                
        if tag in self.tag_cnt:
            self.tag_cnt[tag]+=1
        else:
            self.tag_cnt[tag]=1

            
    def calc_trans(self,tag_cur, tag_nex):
        if tag_cur in self.trans:
            if tag_nex in self.trans[tag_cur]:
                self.trans[tag_cur][tag_nex]+=1
            else:
                self.trans[tag_cur][tag_nex]=1
        else:
            self.trans[tag_cur]={tag_nex:1}

            
    def calc_trans2(self,cur_tag,prev_tag_combo):
        if prev_tag_combo in self.tag_combo_count:
            self.tag_combo_count[prev_tag_combo]+=1
        else:
            self.tag_combo_count[prev_tag_combo]=1
        if cur_tag in self.trans2:
            if prev_tag_combo in self.trans2[cur_tag]:
                self.trans2[cur_tag][prev_tag_combo]+=1
            else:
                self.trans2[cur_tag][prev_tag_combo]=1
        else:
            self.trans2[cur_tag]={prev_tag_combo:1}
        
        
    def word_count_tag_prev(self,cur_word,cur_tag,prev_tag):
        combination=prev_tag+"<>"+cur_tag
        if combination in self.trans_prev_cur:
            self.trans_prev_cur[combination]+=1
        else:
            self.trans_prev_cur[combination]=1
            
        if cur_word in self.word_count_prevtag:
            if combination in self.word_count_prevtag[cur_word]:
                self.word_count_prevtag[cur_word][combination]+=1
            else:
                self.word_count_prevtag[cur_word][combination]=1
        else:
            self.word_count_prevtag[cur_word]={combination:1}
 
# ****************************************************************************************************************************************************************
#Below function is used to compute probabilities for counts calculated above for training.
# Calculation done for below cases:
    
    #1a. Emission probability for a particular state. probability of the word occuring in it's current state. Given by Dict self.word_prob
    #1b. Emission probability for 2 states 1 word. Probability of that word occuring in a state given previous state. Given by Dict self.word_count_prev_prob
    #2.  Probability of tags. Probability of occurence of a particular tag. Given in self.tag_prob
    #3.  Probability of a tag starting the sentence.
    #4.  Probability of multi level self.transition. Given by self.trans2. For Ex: s1,s2,s3 occuring consecutively.
    #5.  Probability of 1 level self.transition. FOr ex: s1,s2 occurring consecutively. Given by self.trans.


# If there is an occurence of a particular self.transition or a particular word, tag pair not being present we give it a minimum value of 10**-10. This is to ensure that duringg Baye's law, this value is not ignored but instead considered very small.
# For MCMC dictionaries of multiple self.transition/emissions, MIN_VALUE is not computed as creating a combination of all 3 level, 2 level combinations of each tag, word is a tedious process. It is better to consider these during testing.
# ****************************************************************************************************************************************************************      
 
    def calc_prob(self,data):
        
        N=len(data)
        total_tags=sum(self.tag_cnt.values())
        pos_tag_list =['noun','adj','verb','.', 'prt','pron', 'det','x','adp','conj','num','adv']
        MIN_VALUE = 10**-10
        
        for word in self.word_prob:
            for tag_prior in pos_tag_list:
                if tag_prior in self.word_prob[word]:
                    self.word_prob[word][tag_prior]/=self.tag_cnt[tag_prior]
                else:
                    self.word_prob[word][tag_prior]=MIN_VALUE
            
        for t_c in self.tag_cnt:
            self.tag_prob[t_c]=self.tag_cnt[t_c]/total_tags
            
        for in_tag in self.initial:
            self.initial_prob[in_tag]=self.initial[in_tag]/N
            
        for w in self.word_count_prevtag:
            self.word_count_prev_prob[w]={}
            for comb in self.word_count_prevtag[w]:
                total_val=self.trans_prev_cur[comb]
                
                self.word_count_prev_prob[w][comb]=self.word_count_prevtag[w][comb]/total_val
                
                
        for word in self.trans2:
            for comb in self.trans2[word]:
                self.trans2[word][comb]/=self.tag_combo_count[comb]
        

        
        for tag_curr in self.trans:
            total_tag_val = sum(self.trans[tag_curr].values())
            for tag_next in pos_tag_list:
                if tag_next in self.trans[tag_curr]:
                    
                    self.trans[tag_curr][tag_next]/=total_tag_val
                else:
                    self.trans[tag_curr][tag_next]= MIN_VALUE
       


# ****************************************************************************************************************************************************************
# Below function is used to calculate log posterior probabilities. 
# For each model, we compute the values based on the interdependence of each node/position.
# If the given word of test is not available in train, we assign it with the value of probability of the tag that occurs the most in Naive Bayes. 
    
    def posterior(self, model, sentence, label):
        MIN_VALUE = 10**-10
        if model == "Simple":
            posterior_val=0
            
            for i in range(len(sentence)):
                word=sentence[i]
                tag_val=label[i]
                if word in self.bag_of_words:
                    if tag_val in self.bag_of_words[word]:
                        posterior_val += math.log(self.word_prob[word][tag_val]*self.tag_prob[tag_val])
                else:
                    highest_tag = max(self.tag_cnt, key=self.tag_cnt.get)
                    posterior_val+=math.log(self.tag_prob[highest_tag])
                    
            return posterior_val
            
            
        elif model == "HMM":
            posterior_val=0
            for i in range(len(sentence)):
                word=sentence[i]
                tag_val=label[i]
                tag_prb=self.tag_prob[tag_val]
                if i<len(sentence)-1:
                        next_tag=label[i+1]
                        posterior_val+=math.log(tag_prb* self.trans[tag_val][next_tag])
                if i == 0:
                    if word in self.word_prob:
                        posterior_val+=math.log(self.initial[tag_val] * self.word_prob[word][tag_val])
                    else:
                        posterior_val+=math.log(MIN_VALUE)
                else:
                    if word in self.word_prob:
                        posterior_val+=math.log(self.word_prob[word][tag_val])
                    else:
                        posterior_val+=math.log(MIN_VALUE)
            return posterior_val
        
        elif model == "Complex":
            posterior_val = 0
            N=len(sentence)
            for i in range(len(sentence)):
                word=sentence[i]
                tag_val=label[i]
                
                if word in self.word_prob[word]:
                    emission1 = math.log(self.word_prob[word][tag_val])
                else:
                    emission1 = math.log(MIN_VALUE)
                
                
                if i<N-1:
                    next_word=sentence[i+1]
                    next_tag=label[i+1]
                    if next_word in self.word_count_prev_prob:
                        if tag_val+"<>"+next_tag in self.word_count_prev_prob[next_word]:
                            prob_emission_fut_state = math.log(self.word_count_prev_prob[next_word][tag_val+"<>"+next_tag])
                        else:
                            prob_emission_fut_state = math.log(MIN_VALUE)
                    else:
                        prob_emission_fut_state = math.log(MIN_VALUE)
                        
                if (i!=0) and (i < N-2):
                    next2_tag=label[i+2]
                    next_tag=label[i+1]
                    if next2_tag in self.trans2:
                        if next_tag+"<>"+tag_val in self.trans2[next2_tag]:
                            prob_trans_fut = math.log(self.trans2[next2_tag][next_tag+"<>"+tag_val])
                        else:
                            prob_trans_fut = math.log(MIN_VALUE)
                    else:
                        prob_trans_fut = math.log(MIN_VALUE)
                else:
                    prob_trans_fut = math.log(MIN_VALUE)
                
                if i!=0 and(i<N-1):
                    next_tag=label[i+1]
                    prob_trans_val = math.log(self.trans[next_tag][tag_val])
                else:
                    prob_trans_val = math.log(MIN_VALUE)
            
            
                if i!=0:
                
                    prev_tag=label[i-1]
                    if word in self.word_count_prev_prob:
                        if prev_tag+"<>"+tag_val in self.word_count_prev_prob[word]:
                            prob_emission_prev_state = math.log(self.word_count_prev_prob[word][prev_tag+"<>"+tag_val])
                        else:
                            prob_emission_prev_state = math.log(MIN_VALUE)
                    else:
                        prob_emission_prev_state = math.log(MIN_VALUE)
                    
            
                
                if i>1:
                    prev_tag=label[i-1]
                    prev2_tag=label[i-2]
                    if tag_val in self.trans2:
                    
                        if prev_tag+"<>"+prev2_tag in self.trans2[tag_val]:
                            prob_trans_prev = math.log(self.trans2[tag_val][prev_tag+"<>"+prev2_tag])
                        else:
                            prob_trans_prev = math.log(MIN_VALUE)
                    else:
                        prob_trans_prev = math.log(MIN_VALUE)
                
                    
                if i == 0:
                    posterior_val+=math.log(self.initial_prob[tag_val])+emission1
                    
                elif i == 1:
                    posterior_val+=emission1+prob_emission_fut_state+prob_trans_fut+prob_trans_val
                elif i==N-2:
                    posterior_val+= prob_trans_prev + prob_trans_val + emission1 + math.log(self.trans[label[i-1]][tag_val]) + prob_emission_prev_state
                elif i==N-1:
                    posterior_val+=prob_trans_prev+prob_trans_val+emission1+math.log(self.trans[label[i-1]][tag_val])
                else:
                    posterior_val+=prob_trans_fut+prob_emission_prev_state+emission1+prob_trans_val+math.log(self.trans[label[i-1]][tag_val]) +prob_emission_fut_state +  prob_emission_prev_state
                
                
            return posterior_val
        else:
            print("Unknown algo!")


# ****************************************************************************************************************************************************************
## calls both counting and probability calc functions to train the given input training file

# ****************************************************************************************************************************************************************

    def train(self, data):
        self.counting(data)
        self.calc_prob(data)


# ****************************************************************************************************************************************************************
# Simplified function computes Naive Bayes where each state and word is independed of consecutive states and words.
# For bc.test: Word Accuracy: 93.92% Sentence Accuracy: 47.45%
# For bc.test.tiny: Word Accuracy: 97.62% Sentence Accuracy:66.67%

# ****************************************************************************************************************************************************************
    def simplified(self, sentence):
        predicted_tag=[]

        for i in range(len(sentence)):
            word=sentence[i]
            
            if word in self.bag_of_words:
                tot_val=sum(self.bag_of_words[word].values())
                pred_prob = {tag : self.bag_of_words[word][tag]/tot_val for tag in self.bag_of_words[word]}               
                pred=max(pred_prob,key=pred_prob.get)   
                predicted_tag.append(pred)
            else:
                predicted_tag.append(max(self.tag_cnt,key=self.tag_cnt.get))
            
        return predicted_tag

# ****************************************************************************************************************************************************************
# HMM Viterbi computes a Viterbi sequence by calculating forward probabilities in the Viterbi table and then backtracking to predict the tags.
# Used Viterbi code provided by Prof. Crandall in class for logic/structure. Other online materials referenced given at beginning of code.
# For bc.test: Word Accuracy: 94.86% Sentence Accuracy: 52.90%
# For bc.test.tiny: Word Accuracy: 97.62% Sentence Accuracy:66.67%

# ****************************************************************************************************************************************************************


    def hmm_viterbi(self, sentence):
        pos_tag_list =['noun','adj','verb','.', 'prt','pron', 'det','x','adp','conj','num','adv']
        V_table={}
        N=len(sentence)
        MIN_VALUE=10**-10
        which_table={}
        for i in range(N):
            word=sentence[i]
            if word not in self.word_prob:
                self.word_prob[word]={}
                for t in pos_tag_list:
                    self.word_prob[word][t]=MIN_VALUE
        
        V_table = {t : [0]*N for t in pos_tag_list}
        which_table = deepcopy(V_table)
   
        for t in pos_tag_list:
            V_table[t][0]=self.initial_prob[t]*self.word_prob[sentence[0]][t]
         
        for i in range(1,N):
            for t in pos_tag_list:
                temp={}
                V_table[t][i] = self.word_prob[sentence[i]][t]

                for t1 in pos_tag_list:
                    
                    temp[t1]=self.trans[t1][t]*V_table[t1][i-1]

                ## TODO: 
                #temp = { t1 : self.trans[t1][t]*V_table[t1][i-1] for t1 in pos_tag_list}
                tag=max(temp,key=temp.get)
                
                which_table[t][i]=tag
                V_table[t][i]*=V_table[tag][i-1]*self.trans[tag][t]
        
        viterbi_seq = [""] * N
        
        ## TODO : 
        temp2 = {}
        for t in pos_tag_list:
            temp2[t]=V_table[t][i]
        
        temp2={ t : V_table[t][i] for t in pos_tag_list}
        
        tag_predicted = max(temp2,key=temp2.get)
        viterbi_seq[N-1]=tag_predicted
        
        for i in range(N-2, -1, -1):
            viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
        return viterbi_seq
    
# ****************************************************************************************************************************************************************
# MCMC Complex function predicts the tags by calculating the probabilities based on given MCMC where there are 
# For bc.test: Word Accuracy: 93.92% Sentence Accuracy: 47.45%
# For bc.test.tiny: Word Accuracy: 97.62% Sentence Accuracy:66.67%

# Number of iterations were self.initially kept to 1000. It took more than 27 mins to run bc.test. Reduced it to 500 and used output of Viterbi as the self.initial
# random sample. Even this took slightly above 10mins. Kept Number of iterations to 150. This is still giving a good score as inital sample is good and 
# final output is converging for most cases under 150.

# References given at top of the code.

# ****************************************************************************************************************************************************************    
    def gibbs_sampling(self,sentence,iterations, initial_sample, pos_tag_list):
        samples_list=[self.initial_sample]
        MIN_VALUE = 10**-10
        
        # Iterate over number of samples you want to generate
        for _ in range(iterations):
            new_sample=[]
            sample=samples_list[-1]
            
            # Iterate over the entire sentence
            for j in range(len(sentence)):
                curr_word=sentence[j]
                if j<len(sentence)-1:
                    next_word=sentence[j+1]
                
                prob={}
                
                # Iterate over all possible_tags in that particular word of the sentence.
                for tag in pos_tag_list:
                    
                    #Calculate self.initial probability. Will be used in cases where that tag begins the sentence.
                    prob_init = self.initial_prob[tag]
                    
                    
                    # Calculate emission probabililty for the current state and current word combo
                    if curr_word in self.word_prob:
                        prob_emission_same_state = self.word_prob[curr_word][tag]
                    else:
                        prob_emission_same_state = MIN_VALUE
                    
                    
                    # Calculate emission based on current state, future state and future word. As s2 is connected to w3. We need to consider impact of s2 while sampling for s2.
                    if j<len(sentence)-1:
                        if next_word in self.word_count_prev_prob:
                            if tag+"<>"+sample[j+1] in self.word_count_prev_prob[next_word]:
                                prob_emission_fut_state = self.word_count_prev_prob[next_word][tag+"<>"+sample[j+1]]
                            else:
                                prob_emission_fut_state = MIN_VALUE
                        else:
                            prob_emission_fut_state = MIN_VALUE
                    

                    # Calculate 2 level future self.transitions for all middle states. s3 is connected to s4 and s5. So while computing samples for s3, we consider it's impact on s4 and s5 as well.        
                    if (j!=0) and (j < len(sentence)-2):
                        if sample[j+2] in self.trans2:                            
                            if sample[j+1]+"<>"+tag in self.trans2[sample[j+2]]:
                                prob_trans_fut = self.trans2[sample[j+2]][sample[j+1]+"<>"+tag]
                            else:
                                prob_trans_fut = MIN_VALUE
                        else:
                            prob_trans_fut = MIN_VALUE
                    else:
                        prob_trans_fut = MIN_VALUE
                        
                    
                    # Calculate immediate self.transition between consecutive nodes. 
                    # As s3 is connected to s2, we need to work on s3 as well while checking for s2.
                    if j!=0 and(j<len(sentence)-1):
                        prob_trans_val = self.trans[sample[j+1]][tag]
                    else:
                        prob_trans_val = MIN_VALUE
                     
                    # Calculate emission based on current state, previous state and current word. 
                    # As w2 is connected to s2 as well as s1, we need to consider impact of previous state as well.
                    if j!=0:
                        if curr_word in self.word_count_prev_prob:
                            if new_sample[j-1]+"<>"+tag in self.word_count_prev_prob[curr_word]:
                                prob_emission_prev_state = self.word_count_prev_prob[curr_word][new_sample[j-1]+"<>"+tag]
                            else:
                                prob_emission_prev_state = MIN_VALUE
                        else:
                            prob_emission_prev_state = MIN_VALUE
                            
                    
                    # As we have computed 2 levels in future to compute multi level self.transition probabilities, 
                    # we also take into account 2 past states. s3 is connected via s2 and s1 as well.  
                    if j>1: 
                       
                        if tag in self.trans2:
                            
                            if new_sample[j-1]+"<>"+new_sample[j-2] in self.trans2[tag]:
                                prob_trans_prev = self.trans2[tag][new_sample[j-1]+"<>"+new_sample[j-2]]
                            else:
                                prob_trans_prev = MIN_VALUE
                        else:
                            prob_trans_prev = MIN_VALUE
                    
                    
                    
                    # Based on all above probabilities each position of the sentence will have different probability calculations. As multiple small values are being considered, we take sum(log(probabilities)) rather than multiplying them to avoid underflow error as well as speed up computation.       
                    if j == 0:
                        
                        prob_val = math.log(prob_init) + math.log(prob_emission_same_state) 
                    elif j == 1:
                         
                        prob_val = math.log(prob_emission_fut_state) +math.log(prob_trans_fut) +math.log(prob_emission_prev_state) +math.log(self.trans[new_sample[j-1]][tag])+ math.log(prob_emission_same_state)+ math.log(prob_trans_val)
                    
                    elif j == len(sentence)-2:
                        
                        prob_val = math.log(prob_trans_prev) + math.log(prob_trans_val) +  math.log(prob_emission_same_state)  + math.log(self.trans[new_sample[j-1]][tag]) + math.log(prob_emission_prev_state) 
                    
                    elif j == len(sentence)-1:
                        
                        prob_val = math.log(prob_trans_prev)+  math.log(prob_emission_same_state) +math.log(self.trans[new_sample[j-1]][tag]) #prob_emission_prev_state *
                    
                    else:
                        prob_val = math.log(prob_trans_fut) +math.log(prob_trans_prev)+math.log(prob_emission_same_state) +math.log(prob_trans_val) +math.log(self.trans[new_sample[j-1]][tag]) +math.log(prob_emission_fut_state) +  math.log(prob_emission_prev_state) 
                    
                    
                    #After all values are calculated we take an exponent to find actual probability and assign to a tag.
                    final_prob=math.exp(prob_val)
                    prob[tag]=final_prob
                    
                
                # Got error when doing random sampling that sum(all prob)!=1. Had to look up normalizing probabilities. 
                # Reference https://newbedev.com/np-random-choice-probabilities-do-not-sum-to-1 For normalizing probabilities.
                sum_p=sum(prob.values())
                for key in prob:
                    prob[key]/=sum_p
                    
                # Referred to MCMC code given by professor Crandall in in-class activity for below part.
                dist=list(prob.items())
                p_val=[dist[i][1] for i in range(len(dist))]
                tags_pos = [dist[i][0] for i in range(len(dist))]
   
                tag_val= np.random.choice(tags_pos,1,p=p_val)
                new_sample.append(tag_val[0])
                
            samples_list.append(new_sample)
        return samples_list
    
    def complex_mcmc(self, sentence):
        
        iterations=175
        self.initial_sample=['noun'] * len(sentence) #self.hmm_viterbi(sentence)
        
        pos_tag_list =['noun','adj','verb','.', 'prt','pron', 'det','x','adp','conj','num','adv']
        samples_list=self.gibbs_sampling(sentence, iterations, self.initial_sample, pos_tag_list)
        
        # In order to find the value of the tag being repeated the most in a particular sentence position, we need to first create a dictionary of position:{tag:count}. Based on the max valued tag we predict that tag as final_tag
        final_count={}
        for i in range(len(samples_list)):
            samples=samples_list[i]
        
            for j in range(len(samples)):
                tag_val=samples[j]
                if j in final_count:
                    if tag_val in final_count[j]:
                        final_count[j][tag_val]+=1
                    else:
                        final_count[j][tag_val]=1
                else:
                    final_count[j]={}
        
                    
        
        predicted_list=[]
        for i in final_count:
            count_vals=final_count[i]
            max_repeated_term=max(count_vals,key=count_vals.get)
            predicted_list.append(max_repeated_term)

        
        return predicted_list


# Final Solve that calls each model to predict results of tags.
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

