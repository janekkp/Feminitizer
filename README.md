# Feminitizer
Deep Learning model that produces feminative of given word.
#

In polish many words have few genders e.g. professions or nationalities, however feminatives (female forms) are often disregarded and male form is used even while talking about women. Many polish have difficulties increating female form and consider them funny or weird. I've tried to build a model that can create a female derivative of a given word (not only profesion or nationality, but any word). Model was trained and evaluated only on suffix derivatives (e.g. lekarz - lekarka), which contribute about 75% of all feminatives in polish (others are e.g. paradigmatic : ojciec - matka). However keeping in mind comlexity of polish language the results are quite good (~ 85% accuracy on test set).

# Development

Data used for this project comes from polish Wordnet (http://plwordnet.pwr.wroc.pl/wordnet ) and from polish dictionary dump (https://sjp.pl/slownik/zg.phtml). 
Model architecture is based on character-level Encoder-Decoder model which is the best know as machine translation example (https://keras.io/examples/nlp/lstm_seq2seq/), however there are some diffrences. 

# Files
Model developmnet is a Python Notebook file which can be used to prepare data, but also to build, train and evaluate model.
Flask App folder consists files required to deploy Flask app which allows to feminitize custom words.

# Future work

In future I'd like to deploy Flask App to some server to allow people play with feminatives in browser. Also I'd like to try to train my own flexer basing on Feminitizer's architecture and what I have learned about character-level Encoder-Decoder. I'm also intresed how this kind of model would deal with similiar task in other languages.
