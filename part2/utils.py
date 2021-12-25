from numpy import *
from scipy.stats import norm
import numpy as np

def calc_prob_dist(x, mean = 0, sd = 1):
    return norm.pdf(x,mean,sd)

def calc_emission_prob(edge_strength, airice_hmm , point):
    #Implementation taken from https://stackoverflow.com/questions/43644320/how-to-make-numpy-array-column-sum-up-to-1
    es = edge_strength.copy()
    if airice_hmm != None:
        for col,each in enumerate(airice_hmm):
            es[0:each+9,col] = 0
    
    em = es / es.sum(axis=0)
    if point != None:
        em[:,point[0]] = 0
        em[point[1],point[0]] = 1 
    return em

def calc_transition_prob(distribution, o, s_prev, s, point):
    #returns the transition probability from previous state to next state
    if point != None:
        if o == point[0]:
            if s == point[1]:
                return 1
            else:
                return 0
    
    diff = s - s_prev
    size_factor = int(len(distribution)/2)
    
    if  (diff + size_factor < len(distribution)) and (diff + size_factor >= 0):
        return distribution[s - s_prev + size_factor] 
    else:
        return 0

def calc_initial_prob(edge_strength,mode="air_ice"):
    edge_strength_normalised = edge_strength / (amax(edge_strength)) 
    if mode == "air_ice":
        return [1 if i==argmax(edge_strength_normalised[:,0]) else 0 for i in range(edge_strength.shape[0])]
    elif mode == "ice_rock":
        return [1 if i==argmax(edge_strength_normalised[argmax(edge_strength_normalised[:,0])+10:,0]) + argmax(edge_strength_normalised[:,0]) + 10 else 0 for i in range(edge_strength.shape[0])]
    else:
        raise Exception("Unknown Mode")

def max_k(o, s, N, trellis, pixel_distribution, point):
    #returns the most likely previous state
    possible_prev_states = [s+i for i in [-2,-1,0,1,2] if ((s+i >= 0) & (s+i < N))]
    max_val = 0
    k = None
    for each_possibility in possible_prev_states:
        calc = trellis[each_possibility,o-1] * calc_transition_prob(pixel_distribution, o, each_possibility, s, point)
        if max_val < calc:
            max_val = calc
            k = each_possibility

    return k

def viterbi(O, S, init_prob, pixel_distribution, Em, point):
    #the pseudo code of this algorithm is referenced from https://en.wikipedia.org/wiki/Viterbi_algorithm
    N = len(S)
    trellis = np.zeros((len(S),len(O)))
    pointers = np.zeros((len(S),len(O)))

    #At Time Zero 
    for s in range(len(S)):
        trellis [s,0] = init_prob[s] * Em[s,0]
    
    trellis[:,0] = (trellis[:,0] / trellis[:,0].sum(axis=0))

    for o in range(1, len(O)):
        for s in range(len(S)):
            k = max_k(o, s, N, trellis, pixel_distribution, point)
            if k != None:
                trellis[s,o] = trellis[k, o-1] * calc_transition_prob(pixel_distribution, o, k, s, point) * Em[s,o]
                pointers[s,o] = k
        trellis[:,o] = (trellis[:,o] / trellis[:,o].sum(axis=0))

    best_path = []
    k = argmax(trellis[:, len(O)-1])
    for o in range(len(O)-1, -1, -1):     # Backtrack from last observation.
        best_path.insert(0, S[k])         # Insert previous state on most likely path
        k = int(pointers[k, o])                # Use backpointer to find best previous state
    return best_path

def initiateViterbi(edge_strength, mode, airice_hmm, point):
    O = [i for i in range(edge_strength.shape[1])]
    S = [i for i in range(edge_strength.shape[0])]
    init_prob = calc_initial_prob(edge_strength, mode)
    pixel_distribution = calc_prob_dist(np.arange(-2, 3, 1), mean = 0, sd = 1)
    Em = calc_emission_prob(edge_strength, airice_hmm, point)
    
    return viterbi(O, S, init_prob, pixel_distribution, Em, point)