# Experiments in word category creation with transformers
###### Journal by glicerico

## Jan 2020
We start experimenting with BERT, since it's what people were using in their 
word sense disambiguation experiments (Wiedemman et al. 2019).

The general idea here is that BERT (and other transformer models) have shown 
great language abstraction capabilities.
However, those abstractions don't seem to be found in the attention layers in
a straightforward manner (see Clarke et al. 2019, Htut et al. 2019), and not
easily extractable in an unsupervised manner (for supervised mapping of the
syntactic knowledge learned by BERT, see Hewitt and Manning 2019).
Thus, Ben proposes to use external statistical methods to "milk" the syntactic
relationships that we are looking for out of BERT.

Based on the code by Wiedemann et al. (2019, github.com/uhh-lt/bert-sense), I 
first updated it to use the latest huggingface transformers models 
(github.com/huggingface/transformers) and to be able to run without CUDA 
(fixed a bug).
The obtained results were not exactly the same as their paper, but quite close.
I blame the change of transformers versions for the slight difference.

Then, I wanted to test my idea of just getting embeddings for every word
in the corpus, then cluster them and hope that word categories will appear
magically.
this idea is coming from the previous use of word2vec and AdaGram vectors
for word categorization. However, at least two problems were noted:
1) Semantic relatedness seems to be an important component of the
embeddings here, while syntactic functions were not clearly distinguished in 
the resulting clusters: words like medicine, medicinal, pharmacist would 
commonly fall in the same cluster.
2) Memory requirements grew very quickly when handling a decent corpus,
since a unique word embedding needs to be stored for each single word 
in the corpus.

So, I decided to go back to the plan discussed with Ben last week, which
is more or less like this:

1) 

