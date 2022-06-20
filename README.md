# Limitations of the entropy measure in n-gram language modelling (CLIN32)
### by Michael Bauwens, Peter Vanbrabant, José Tummers (UCLL Research & Expertise - Smart Organisations)

This demo accompanies the poster presentation "Limitations of the entropy measure in n-gram language modelling". It provides the functionality to:
- import data (default: Jane Austen's "Emma")
- train a trigram language model on this dataset
- compute the probabilities of every sentence in the dataset
- score every sentence with a variety of entropy measures
- explore the correlations between entropy measures
- retrieve the most probable sentences (low entropy) based on every measure

This repository contains a Jupyter Notebook with the main flow, a paired Python file, and a Python file (`tools.py`) with the functions used in the notebook. Additionally, if you work with Poetry, you can use the dependency files (`poetry.lock` and `pyproject.toml`) to synchronise your Python 3.10 interpreter.

Check it out on [Google Colab](https://colab.research.google.com/drive/1NtwOvY58SYT7YkIwEFMm2_XBR1zMJp7U?usp=sharing)

In the `/poster_and_abstract` folder, you'll find (as expected) the poster and abstract which were presented on CLIN32.

- [Abstract on CLIN32 website](https://clin2022.uvt.nl/limitations-of-the-entropy-measure-in-n-gram-language-modelling/)
- [UCLL Research & Expertise](https://research-expertise.ucll.be/)
