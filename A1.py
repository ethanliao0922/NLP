import sys
import math
from collections import defaultdict, Counter

def read_file(path):
    f = open(path, "r")
    lst = []
    for line in f:
        lst.append(line)

    return lst

def tokenize(corpus):
    return [sentence.split() for sentence in corpus]

def count_amt(word_lst):
    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter()
    for sentence in word_lst:
        for i in range(len(sentence) - 1):
            unigram_counts[sentence[i]] += 1
            bigram_counts[sentence[i]][sentence[i + 1]] += 1
        
        unigram_counts[sentence[-1]] += 1 # Count the last word

    return unigram_counts, bigram_counts

def cal_prob(uni_cnt, bi_cnt):
    uni_freq, bi_freq = dict(), defaultdict(dict)
    uni_sum = sum(uni_cnt.values())
    for token in uni_cnt:
        cnt = uni_cnt[token]
        uni_freq[token] = cnt / uni_sum

    for a in bi_cnt:
        for b in bi_cnt[a]:
            bi_sum = uni_cnt[a]
            cnt = bi_cnt[a][b]
            bi_freq[a][b] = cnt / bi_sum

    return uni_freq, bi_freq

def replace_with_UNK(tokenized_corpus, word_counts, k):
    updated_corpus = []
    for sentence in tokenized_corpus:
        updated_sentence = [word if word_counts[word] >= k else "<UNK>" for word in sentence]
        updated_corpus.append(updated_sentence)

    return updated_corpus

def add_k(uni_cnt, bi_cnt, k):
    uni_freq, bi_freq = dict(), defaultdict(dict)
    v = len(uni_cnt)
    uni_sum = sum(uni_cnt.values()) + (k*v)
    for token in uni_cnt:
        cnt = uni_cnt[token] + k
        uni_freq[token] = cnt / uni_sum

    for a in bi_cnt:
        bi_sum = uni_cnt[a] + (k*v)
        for b in bi_cnt[a]:
            cnt = bi_cnt[a][b] + k
            bi_freq[a][b] = cnt / bi_sum

    return uni_freq, bi_freq

def linear_interpolation(uni_cnt, bi_cnt, l1, l2):
    uni_prob, bi_prob = dict(), defaultdict(dict)
    uni_sum = sum(uni_cnt.values())
    for token in uni_cnt:
        cnt = uni_cnt[token]
        uni_prob[token] = cnt / uni_sum

    for a in bi_cnt:
        bi_sum = uni_cnt[a]
        for b in bi_cnt[a]:
            bi_prob[a][b] = (l1 * (bi_cnt[a][b] / bi_sum)) + (l2 * uni_prob.get(b, 0))

    return uni_prob, bi_prob

def replace_unseen_words(validation_set, word_counts, k):
    updated_validation = []
    for sentence in validation_set:
        updated_sentence = [word if word in word_counts and word_counts[word] >= k else "<UNK>" for word in sentence]
        updated_validation.append(updated_sentence)

    return updated_validation

def cal_sentence_prob_add_k(sentence, bigram_probabilities, unigram_counts, v):
    prob = 1.0
    for i in range(len(sentence) - 1):
        w1, w2 = sentence[i], sentence[i + 1]
        prob *= bigram_probabilities.get(w1, {}).get(w2, 1 / (unigram_counts[w1] + v))  # Apply smoothing
        if prob == 0.0: # too small to store
            return sys.float_info.min # smallest float
        
    return prob

def cal_sentence_prob_linear(sentence, bigram_probabilities, unigram_probabilities, lambda1, lambda2):
    prob = 1.0
    for i in range(len(sentence) - 1):
        w1, w2 = sentence[i], sentence[i + 1]
        bi_prob = bigram_probabilities.get(w1, {}).get(w2, 0)
        uni_prob = unigram_probabilities.get(w2, 0)
        interpolated_prob = lambda1 * bi_prob + lambda2 * uni_prob
        prob *= interpolated_prob 
        if prob == 0.0: 
            return sys.float_info.min # avoid zero probability
    
    return prob

def cal_perplexity(validation_set, bigram_probabilities, unigram_counts, m, unigram_probabilities, lambda1, lambda2):
    v = len(unigram_counts)
    n = sum(len(sentence) - 1 for sentence in validation_set)  # Total number of bigrams
    log_sum = 0
    
    for sentence in validation_set:
        if m == 1:
            sentence_prob = cal_sentence_prob_add_k(sentence, bigram_probabilities, unigram_counts, v)
        elif m == 2:
            sentence_prob = cal_sentence_prob_linear(sentence, bigram_probabilities, unigram_probabilities, lambda1, lambda2)

        log_sum += -math.log(sentence_prob, 2)
    
    return math.exp(log_sum / n)

if __name__ == "__main__":
    train = read_file("train.txt")
    tokenize_train = tokenize(train)
    uni_cnt, bi_cnt = count_amt(tokenize_train)

    # unsmmothing
    uni_prob, bi_prob = cal_prob(uni_cnt, bi_cnt)

    # unknown words
    train_with_unk = replace_with_UNK(tokenize_train, uni_cnt, 2)
    uni_cnt_with_ukn, bi_cnt_with_ukn = count_amt(train_with_unk)

    # smoothing 1: add-k
    add_k_uni_prob, add_k_bi_prob = add_k(uni_cnt_with_ukn, bi_cnt_with_ukn, 2)

    # smoothing 2: linear interpolation
    lambda1, lambda2 = 0.3, 0.7
    li_uni_prob, li_bi_prob = linear_interpolation(uni_cnt_with_ukn, bi_cnt_with_ukn, lambda1, lambda2)

    # validation
    val = read_file("val.txt")
    val_with_unk = replace_unseen_words(tokenize(val), uni_cnt, 2)
    
    s1_perplexity_value = cal_perplexity(val_with_unk, add_k_bi_prob, uni_cnt_with_ukn, 1, add_k_uni_prob, lambda1, lambda2)
    print(f"Perplexity on the validation set with add_k smoothing: {s1_perplexity_value}")

    s2_perplexity_value = cal_perplexity(val_with_unk, li_bi_prob, uni_cnt_with_ukn, 2, li_uni_prob, lambda1, lambda2)
    print(f"Perplexity on the validation set with linear interpolation smoothing: {s2_perplexity_value}")
