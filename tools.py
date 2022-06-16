import os
from nltk import FreqDist, trigrams
from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from numpy import log2, mean, random, array
import seaborn as sns
from scipy import stats

def import_data(datadir, lang):
    if lang == 'nl':
        from spacy.lang.nl import Dutch
        nlp = Dutch()
    elif lang == 'en':
        from spacy.lang.en import English
        nlp = English()
    else:
        print(f"{lang} not configured in the function."
               f"Adapt the function to include the appropriate language (if supported by spacy).")
    nlp.add_pipe('sentencizer')
    for file in os.listdir(datadir):
        with open(datadir+file, 'r') as f:
            text = f.read()
            doc = nlp(text)
            text = [[w.lower_ for w in s if not w.is_space] for s in doc.sents]
            text = [x for x in text if x]  # remove empty sentence lists
            data += text
    return data

def train_lm(corpus, cutoff=2):
    train, vocab = padded_everygram_pipeline(3, corpus)
    
    if cutoff > 1:
        vocab = Vocabulary(vocab, unk_cutoff=cutoff)

    lm = MLE(3)
    lm.fit(train, vocab)
    
    print(f"With cutoff {cutoff}, the amount of <UNK> tokens is {lm.counts.unigrams['<UNK>']}.")

    return lm

def get_trigrams(sentence):
    # the sentence should be formatted as a list of words
    return trigrams(pad_both_ends(sentence, 3))

def populate_dicts(corpus, lm):
    fdist = FreqDist()
    sent_dict = {}
    for i, sent in enumerate(corpus):
        tri = list(get_trigrams(sent))
        prob_list = []
        for trigram in tri:
            score = lm.score(trigram[-1], trigram[:-1])
            prob_list.append(score)
            fdist[' '.join(trigram)] += 1

        sent_dict[i] = {'text': ' '.join(sent),
                        'trigrams': tri,
                        'prob_list': prob_list}

    fdist = {'fd': fdist,
         'total': fdist.total(),
         'rel_fd': {}}

    for key in fdist['fd'].keys():
        fdist['rel_fd'][key] = fdist['fd'][key]/fdist['total']
    
    return fdist, sent_dict

def get_random_sample_prob(sent_dict):
    print("Random sample of a sentence with its probabilities\n")
    sent_amount = len(sent_dict)
    rand_id = random.randint(0,sent_amount+1)
    print(
        f"Sentence (#{rand_id}): {sent_dict[rand_id]['text']}\n\n"
        f"Trigrams: {sent_dict[rand_id]['trigrams']}\n\n"
        f"Probabilities: {sent_dict[rand_id]['prob_list']}\n\n"
    )

def shannon_entropy(sent_dict_entry):
    """Shannon Entropy: negative sum over all probabilities*log2_probabilities
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    See chapter 3 of Speech and Language Processing (Jurafsky and Martin, 2021) Formula 3.41.
    """
    return -1 * sum([prob * log2(prob) for prob in sent_dict_entry['prob_list']])

def length_normalised_shannon_entropy(sent_dict_entry):
    """Length normalised Shannon Entropy: divide Shannon entropy by the amount of trigrams in the sentence."""
    return shannon_entropy(sent_dict_entry) / len(sent_dict_entry['prob_list'])

def relfreq_weighted_shannon_entropy(sent_dict_entry, fdist, lm, length_normalisation=False):
    sent_dict_entry_cp = sent_dict_entry
    def contains_UNK(tri):
        bool_list = []
        for t in tri.split():
            if lm.counts.unigrams[t] == 0:  # <UNK> token
                bool_list.append(True)
            else:
                bool_list.append(False)
        return any(bool_list)

    weighted_prob_list = []
    for prob, tri in zip(sent_dict_entry['prob_list'], sent_dict_entry['trigrams']):
        if contains_UNK(tri):
            prob *= min(fdist['rel_fd'].values())
        else:
            prob *= fdist['rel_fd'][tri]
        weighted_prob_list.append(prob)
    entropy = -1 * sum([prob * log2(prob) for prob in weighted_prob_list])
    if length_normalisation:
        entropy /= len(sent_dict_entry['prob_list'])
    return entropy

def shannon_mcmillan_breiman_entropy(sent_dict_entry):
    """From nltk.lm: negative average log2_probabilities
    https://www.nltk.org/api/nltk.lm.api.html#nltk.lm.api.LanguageModel.entropy
    See chapter 3 of Speech and Language Processing (Jurafsky and Martin, 2021) Formula 3.47.
    """
    return -1 * mean([log2(prob) for prob in sent_dict_entry['prob_list']])

def entropy_measures(sent_dict_entry, fdist, lm, verbose=False):
    shan = shannon_entropy(sent_dict_entry)
    shan_lengthnorm = length_normalised_shannon_entropy(sent_dict_entry)
    shan_mcmill_brei = shannon_mcmillan_breiman_entropy(sent_dict_entry)
    shan_relfreqweight = relfreq_weighted_shannon_entropy(sent_dict_entry, fdist, lm)
    shan_length_relfreq = relfreq_weighted_shannon_entropy(sent_dict_entry, fdist, lm, length_normalisation=True)
    
    sent_length = len(sent_dict_entry['prob_list'])
    low_relfreq_perc = len([tri for tri in sent_dict_entry['trigrams'] if fdist['fd'][' '.join(tri)] < 4]) / sent_length
    perfect_prob_perc = len([prob for prob in sent_dict_entry['prob_list'] if prob == 1]) / sent_length
    
    if verbose:
        print(
            f"Sentence: \"{sent_dict_entry['text']}\"\n"
            f"Sentence length (in trigrams): {sent_length}\n"
            f"Percentage of low trigram relfreq (count<4): {round(low_relfreq_perc*100)}%\n"
            f"Percentage of perfect probability: {round(perfect_prob_perc*100)}%\n"
            f"Shannon entropy: {round(shan,4)}\n"
            f"Shannon entropy (length normalised): {round(shan_lengthnorm,4)}\n"
            f"Shannon-McMillan-Breiman entropy (~ length normalised): {round(shan_mcmill_brei,4)}\n"
            f"Shannon entropy (relative frequency weighted): {round(shan_relfreqweight,4)}\n"
            f"Shannon entropy (length normalised + relfreq weighted): {round(shan_length_relfreq,4)}\n"
        )
    
    sent_dict_entry["sent_length"] = sent_length
    sent_dict_entry["low_relfreq_perc"] = low_relfreq_perc
    sent_dict_entry["perfect_prob_perc"] = perfect_prob_perc
    sent_dict_entry["shan"] = shan
    sent_dict_entry["shan_lengthnorm"] = shan_lengthnorm
    sent_dict_entry["shan_mcmill_brei"] = shan_mcmill_brei
    sent_dict_entry["shan_relfreqweight"] = shan_relfreqweight
    sent_dict_entry["shan_length_relfreq"] = shan_length_relfreq
    
    return sent_dict_entry

def get_random_sample_ent(sent_dict, fdist, lm):
    sent_amount = len(sent_dict)
    rand_id = random.randint(0,sent_amount+1)
    sent_dict_entry = sent_dict[rand_id]
    entropy_measures(sent_dict_entry, fdist, lm, verbose=True) # ADJUST TO DF

def get_corr_pairplot(df):
    def corrfunc(x, y, hue=None, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        r, _ = stats.pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f"ρ = {r:.2f}", xy=(.7, .9), xycoords=ax.transAxes)

    p = sns.pairplot(df, color='#002757')
    p.map_lower(corrfunc)

    return p

def get_freq_counts(df, lm, fdist, index):
    tri_list = []
    uni_list = []
    
    def get_tri_counts(t):
        if lm.counts.unigrams[t] == 0:
            uni_list.append(f"{t} (<UNK>={lm.counts.unigrams['<UNK>']})")
        else:
            uni_list.append(f"{t} ({lm.counts.unigrams[t]})")
    
    for i, tri in enumerate(df.filter(items = [index], axis=0)['trigrams'].item()):
        tri = ' '.join(tri)
        tri_list.append(f"{tri} ({fdist['fd'][tri]})")
        if i == 0:
            for t in tri.split():
                get_tri_counts(t)
        else:
            get_tri_counts(tri.split()[2])
          
    print("Trigram counts:")
    print('\n'.join(tri_list))
    print("\nUnigram counts:")
    print('\n'.join(uni_list))
