# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Limitations of the entropy measure in n-gram language modelling (CLIN32)

# ### by Michael Bauwens (UCLL Research & Expertise)

# This demo accompanies the poster presentation "Limitations of the entropy measure in n-gram language modelling". It provides the functionality to:
# - import data (default: Jane Austen's "Emma")
# - train a trigram language model on this dataset
# - compute the probabilities of every sentence in the dataset
# - score every sentence with a variety of entropy measures
# - explore the correlations between entropy measures
# - retrieve the most probable sentences (low entropy) based on every measure

# ## Import required modules

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tools import *
from nltk import download
from nltk.corpus import gutenberg

# ## Import the data

# As toy example, we'll use Jane Austen's "Emma" (provided by the Gutenberg project via NLTK).

download('gutenberg')
data = [[w.lower() for w in s] for s in gutenberg.sents('austen-emma.txt')]

# If you want to try this notebook with your own dataset, you can provide the text files. The function `import_data` _should_ transform the data into the correct format for the language modelling (a list of sentences, each consisting of lists of tokenised words). Currently the function is configured to support Dutch and English via the `lang` parameter. Should you require a different language, you can adapt the function (for language supported in spaCy v3, see [spacy.io/usage/models](https://spacy.io/usage/models)).

# +
# data = import_data("/path/to/directory/", lang="en")
# -

# ## Train the language model

# We'll use a trigram statistical language model, using a vocabulary with cutoff 2. We're using the standard language model and not a model with any smoothing (e.g. Kneser-Ney Interpolated smoothing) because we'll just evaluate the probability of sentences within the corpus *itself*.

lm = train_lm(data, cutoff=2)

# ## Compute probabilities for every sentence

# For every sentence we'll compute the raw probabilities and store them in a dictionary together with the string version of the sentence and a list of the trigrams in the sentence. Additionally, we'll create a frequency distribution of the trigrams in the dataset, which we'll need for the frequency weighting later on. 

fdist, sent_dict = score_sentences(data, lm)

# You can explore the sentence probabilities using the `get_random_sample(sent_dict)` function.

get_random_sample_prob(sent_dict)

# ## Calculate entropy measures

# For each sentence, we add the different entropy measures to the dictionary as well as some additional information (sentence length, percentage of trigrams with low relative frequency, and percentage of trigrams with perfect probability).
#
# Entropy measures:
# - Shannon entropy
# - Length normalised Shannon entropy
# - Shannon entropy via Shannon-McMillan-Breiman theorem
# - Relative frequency weighted Shannon entropy
# - Length normalised relative frequency weighted Shannon entropy
#
# The dictionary is then transformed into a Pandas dataframe.

# +
for sent_id in sent_dict:
    sent_dict[sent_id] = entropy_measures(sent_dict[sent_id], fdist, lm)
    
df = pd.DataFrame.from_dict(sent_dict, orient='index')
# -

# Check out some random samples.

get_random_sample_ent(df)

# Now we'll investigate how the different measures relate to one another in terms of correlation.

get_corr_pairplot(df)

# ### Sentences with lowest entropy

# For Shannon entropy, length normalised Shannon entropy, and length normalised relative frequency weighted Shannon entropy we'll display the 10 sentences with lowest entropy.

df.sort_values(by=['shan']).head(10)

df.sort_values(by=['shan_lengthnorm']).head(10)

df.sort_values(by=['shan_length_relfreq']).head(10)

# If you want to look up the frequency specifics of a certain sentence, you can use `get_freq_counts` to get frequency counts for the tri- and unigrams.

get_freq_counts(df, lm, fdist, 420)

# Finally, we'll visualise the difference between Shannon entropy and length normalised relative frequency weighted Shannon entropy in terms of their correlation.

print(stats.pearsonr(df['shan_length_relfreq'], df['shan']))
plt.scatter(df['shan_length_relfreq'], df['shan'], color='#002757')
plt.xlabel("Length normalised frequency weighted Shannon entropy")
plt.ylabel("Shannon entropy")
