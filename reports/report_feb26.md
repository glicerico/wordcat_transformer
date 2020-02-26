TL;DR: 
I added handling of sub-words, modified the way sentence probabilities
are calculated, and integrated disambiguation into the pipeline.
After these changes, the clusters obtained are still good, but several
words are still uncategorized.
**********
Using [this small corpus](../vocabularies/smallWSD_corpus.vocab), cleaning 5%
of the most common words (assumed as function words, although in this
example this threshold is off), and clustering (KMeans, k=2)
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
[word categorizer](../src/word_categorizer.py), I create a separate
vector for each sense.
Thus I need to fill a matrix of dimensions (nbr of word-senses, 
nbr of sentences).
When evaluating the probability of a given sentence for a word in 
the vocabulary,
I find which of the existing word-sense centroids for that word
(if more than one) is cosine-closest 
to the embedding of the instance of the word in the given sentence,
and assign the sentence probability ONLY to that word vector, leaving
the others as 0.

I get [these](../tests/smallWSD_cat)
categories for the same corpus.
The clusters obtained by the OPTICS algorithm are as good as before, 
but many
words remain uncategorized (cluster #-1, considered 'noise' in the algo).
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
what most other word-embedding algorithms use, whereas K-means cannot in
this implementation.

I didn't notice a relevant improvement adding the word senses, 
and actually none of the disambiguated words were clustered.
This makes me wonder if the disambiguated embeddings are actually correct.
From the WSD part, it seems like they do make sense, otherwise I wouldn't
get so sharp disambiguation. Also, these final layer embeddings is
what others have used for disambiguation (e.g. 
the original BERT paper [Devlin+19], the BERT-sense paper[Wiedeman+19]),
but they use some labeled data in their process.
I averaged the embeddings from all instances in the same sense to get the
centroid, so maybe this is not representing the cluster properly.
Alternatively, maybe dividing each word vector into multiple senses ruins
the vector and becomes so different from the rest that it looks like noise.

Ben has his doubts about these embeddings for unsupervised learning, anyway.
He proposed a different way of getting the embeddings using sentence-probs.
However, if the problem is the latter (vectors being ruined by leaving 
zeros in all cells where they are not the most likely sense), then
a different way of getting the embeddings doesn't help.
Any ideas on how to work with word-senses in this process is welcome.


Checks:
- Ben, I want to make sure that I interpreted your sentence probability estimation
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

Now, because in the above calculation the probability of a sentence is 
inversely 
proportional to its
length (i.e. a perfectly grammatical sentence tends to have a much 
lower score 
the longer it is), I feel it's important to normalize by sentence length.
I have been using the geometric average of each term for this, but I am
not sure of the probabilistic implications of it:
```
P_forward(he answered quickly) = [P(he) * P(he answered | he) * P( he answered quickly | he answered)]^(1/N)
```
where `N=3` in this example, of course.
There is still a lower probability for longer sentences, and it is not as
dramatic as without averaging.
However, it does take the longer sentence down to the probability
of an ungrammatical shorter sentence:
```
Processing sentence: ['[CLS]', 'the', 'test', 'was', 'success', 'a', '.', '[SEP]']

Word: the 	 Prob_forward: 0.006482909899204969; Prob_backwards: 0.7925955057144165
Word: test 	 Prob_forward: 9.367070742882788e-05; Prob_backwards: 0.0005686060176230967
Word: was 	 Prob_forward: 0.3028062582015991; Prob_backwards: 0.001804494415409863
Word: success 	 Prob_forward: 2.0934478015988134e-05; Prob_backwards: 9.219898799983639e-08
Word: a 	 Prob_forward: 6.565311196027324e-05; Prob_backwards: 0.0010389169910922647
Word: . 	 Prob_forward: 0.7495615482330322; Prob_backwards: 0.7197229266166687
Geometric-mean forward sentence probability: 0.010831358316647475
Geometric-mean backward sentence probability: 0.009302214880680615


Average normalized sentence prob: 0.010037710023262371

Processing sentence: ['[CLS]', 'the', 'party', 'was', 'a', 'success', '.', '[SEP]']

Word: the 	 Prob_forward: 0.006482909899204969; Prob_backwards: 0.9431305527687073
Word: party 	 Prob_forward: 0.001439713523723185; Prob_backwards: 0.005618405994027853
Word: was 	 Prob_forward: 0.0562421977519989; Prob_backwards: 0.9035378694534302
Word: a 	 Prob_forward: 0.00046723693958483636; Prob_backwards: 0.3296686112880707
Word: success 	 Prob_forward: 0.467549204826355; Prob_backwards: 2.4018852855078876e-05
Word: . 	 Prob_forward: 0.9529269337654114; Prob_backwards: 0.7197229266166687
Geometric-mean forward sentence probability: 0.05686127007821229
Geometric-mean backward sentence probability: 0.1133680851545998


Average normalized sentence prob: 0.08028856274853516

Processing sentence: ['[CLS]', 'the', 'farewell', 'party', 'was', 'definitely', 'not', 'a', 'success', '.', '[SEP]']

Word: the 	 Prob_forward: 0.00395584711804986; Prob_backwards: 0.8389569520950317
Word: farewell 	 Prob_forward: 4.800636816071346e-06; Prob_backwards: 0.0009747481090016663
Word: party 	 Prob_forward: 0.008732594549655914; Prob_backwards: 0.007712601218372583
Word: was 	 Prob_forward: 0.7753593921661377; Prob_backwards: 0.9089738726615906
Word: definitely 	 Prob_forward: 4.963572337146616e-06; Prob_backwards: 1.5817162193343393e-06
Word: not 	 Prob_forward: 0.03754209354519844; Prob_backwards: 0.016013313084840775
Word: a 	 Prob_forward: 0.11118010431528091; Prob_backwards: 0.01865754835307598
Word: success 	 Prob_forward: 0.12201588600873947; Prob_backwards: 4.0724535210756585e-05
Word: . 	 Prob_forward: 0.9771884679794312; Prob_backwards: 0.7714381814002991
Geometric-mean forward sentence probability: 0.02081403606509604
Geometric-mean backward sentence probability: 0.018465632081794924

Average normalized sentence prob: 0.01960470178593069
```

Also, processing takes very long (processing the 15 sentences in
 smallWSD corpus takes hours in my laptop), 
 which makes it hard to test new improvements.
 UPDATE: Sergey will provide a server where I can run.
 I should also find ways to make the algorithm more efficient.

NEXT:
- Improve running-time efficiency, to try variations faster.
- Try a different way of generating/using sense-centroids
- Try other clustering algorithms with automated cluster number 
detection (Gaussian mixtures?)
