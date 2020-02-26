TL;DR: 
I added handling of sub-words, modified the way sentence probabilities
are calculated, and integrated disambiguation into the pipeline.
After these changes, the clusters obtained are still good, but several
words are still uncategorized.
**********
Using [this small corpus](../vocabularies/smallWSD_corpus.vocab), cleaning 5%
of the most common words (assumed as function words, although in this
example this threshold is off), and clustering with KMeans for 2 clusters
all words with 2 or more instances in the corpus, 
I get [this](../tests/smallWSD_KMeans_k2) disambiguation results.
Looking at the ambiguous words in the corpus (e.g. [dress](../tests/smallWSD_KMeans_k2/dress.disamb),
[mouse](../tests/smallWSD_KMeans_k2/mouse.disamb)),
one can see
consistent good results:
```
Cluster #0:
[(0, 6), (13, 2), ]
Samples:
the brown MOUSE wasn ' t able to feed and dress his growing family last year .
the fat cat ate the last MOUSE quickly .
Cluster #1:
[(2, 1), (14, 5), ]
Samples:
she bought the last microsoft MOUSE for santiago .
my MOUSE stopped responding at the same time as the keyboard .
```
Then, providing cluster centers for each obtained sense there to the
[word categorizer](../src/word_categorizer.py), I get 
[these](../tests/smallWSD_cat)
categories for the same corpus.
The clusters obtained by the OPTICS algorithm as good as before, but many
words are uncategorized.
Setting the minumum cluster size to 2 (lowest possible value), 
[we get](../tests/smallWSD_cat/OPTICS_min_samples_2.wordcat):
```
Cluster #-1:
[fat, fat, ate, last, mouse, mouse, quickly, quickly, ., there, there, many, many, health, health, associated, with, with, stopped, responding, same, time, as, will, fly, fly, negative, of, a, a, long, in, us, tomorrow, she, she, was, was, wearing, lovely, brown, brown, dress, attendees, did, not, properly, for, for, occasion, became, after, got, married, ', ', s, deteriorated, and, de, ,, ago, fitting, wasn, t, year, smith, protagonize, ]
Cluster #0:
[the, my, his, ]
Cluster #1:
[born, able, ]
Cluster #2:
[raised, growing, bought, ]
Cluster #3:
[cat, keyboard, car, session, feed, family, microsoft, ]
Cluster #4:
[duration, episode, series, ]
Cluster #5:
[are, is, ]
Cluster #6:
[morning, night, ]
Cluster #7:
[away, out, ]
Cluster #8:
[they, he, i, you, ]
Cluster #9:
[risks, effects, ]
Cluster #10:
[at, to, ]
Cluster #11:
[santiago, cuba, ]
Cluster #12:
[time, will, long, ]
Cluster #13:
[dress, and, ]
Cluster #14:
[of, in, ]
```
KMeans does a much [poorer job](../tests/smallWSD_cat/KMeans_k_10.wordcat) 
at categorizing these.
One interesting difference is that OPTICS can use cosine distance, which is
what most other word-embedding algorithms use, whereas K-means cannot
(and apparently [should not](), as I am learning).

I didn't notice a relevant improvement in that respect, and actually none
of the disambiguated words were clustered.
This makes me wonder if the disambiguated embeddings are actually correct.
These final layer embeddings is what others have used for disambiguation 
(e.g. 
the original BERT paper [Devlin+19], the BERT-sense paper[Wiedeman+19]),
but they use some labeled data in their process.

Ben has his doubts about these embeddings for unsupervised learning, anyway.


Checks:
- I want to make sure that I interpreted your sentence probability estimation
idea correctly.
You suggested that the probability of a sentence `P(S)` could be calculated as

<code> P(S) = sqrt(P<sub>forward</sub>(S) * P<sub>backward</sub>(S)) </code>

where, e.g.

```
P_forward(he answered quickly) = P(he) * P(he answered | he) * P( he answered quickly | he answered)
```
Because BertForMaskedLM actually gives you the probability of a word
filling in a `[MASK]` space, I am interpreting these probabilities as:
```
P(he) = P(w_0 = he | [MASK] [MASK] [MASK])
P(he answered | he) = P(w_1 = answered | he [MASK] [MASK])
P(he answered quickly | he answered) = P(w_2 = quickly | he answered [MASK])
```
Similarly for the backward pass:
```
P(answered quickly | quickly) = P(w_1 = answered | [MASK] [MASK] quickly)
```

Now, because the probability of a sentence is inversely proportional to its
length (i.e. a perfectly grammatical sentence tends to have a lower score 
the longer it is), I feel it's important to normalize by sentence length.
I have been using the geometric average of each term for this, but I am
not sure of the probabilistic implications of it:
```
P_forward(he answered quickly) = [P(he) * P(he answered | he) * P( he answered quickly | he answered)]^(1/N)
```
where `N=3` in this example, of course.

Problems:
- Processing takes very long, which makes it hard to test new improvement
ideas
- Example

NEXT:
- Try clustering algorithms with automatic cluster detection