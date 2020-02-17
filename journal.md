# Experiments in word category creation with transformers
###### Journal by glicerico

## Jan 2020
Start experiments using BERT, since it's what people were using in their 
word sense disambiguation experiments (Wiedemman et al. 2019).

The general idea here is that BERT (and other transformer models) have shown 
great language abstraction capabilities.
However, those abstractions don't seem to be found in the attention layers in
a straightforward manner (see Clarke et al. 2019, Htut et al. 2019), and not
easily extractable in an unsupervised manner (for supervised mapping of the
syntactic knowledge learned by BERT, see Hewitt and Manning 2019).
Thus, Ben proposes to use external statistical methods to "milk" BERT the syntactic
relationships that we are looking for.

Based on the code by Wiedemann et al. (2019, github.com/uhh-lt/bert-sense), I 
first updated [Bert_Model.py](src/Bert_Model.py) to use the latest huggingface transformers models 
(github.com/huggingface/transformers) and to be able to run without CUDA 
(fixed a bug).
The obtained results were not exactly the same as their paper, but quite close.
I blame the change of transformers versions for the slight difference.

Then, I wanted to test my idea of just getting embeddings for every word
in the corpus, then cluster them and hope that word categories will appear
as a result.
This idea is coming from the previous use of word2vec and AdaGram vectors
for word categorization. 
The code for this attempt is at [all_word_senses.py](src/all_word_senses.py).
Some of the design decisions made were:
- Use the concatenation of the last 4 attention layers for embeddings, 
like bert-sense and the original BERT paper (Devlin et al 2018).
- For words split into sub-components by the BERT tokenizer, use
the arithmetic average of the embeddings of each sub-token, like bert-sense.
- Tried clustering with KMEANS, DBSCAN, OPTICS models in scikit's sklearn
libraries.
- Truncated the precision of the word embeddings to float32 and 5 decimal
places, to try to make processing more efficient. Would be a good idea to
confirm this doesn't affect results.
- Use the cosine distance metric in clustering.

After these attempts, 
at least two problems were noted:
1) Semantic relatedness seems to be an important component of the
embeddings here, while syntactic functions were not clearly distinguished in 
the resulting clusters: words like medicine, medicinal, pharmacist would 
commonly fall in the same cluster.
2) Memory requirements grew very quickly when handling a decent-sized corpus
(8611 sentences, 251,767 words),
since a unique word embedding is required for each single word instance
in the corpus.
**********

Instead, I decided to proceed with the plan discussed with Ben last week, which
goes more or less like this:

1) Disambiguate words using their unique embeddings: 
come up with a few senses for each word above some frequency threshold.
2) Build a matrix of word-sense-pair similarities by calculating the difference
in sentence probabilities between each word and other words.
3) Use above matrix to get word-sense embeddings.
4) Cluster the vectors to form word categories. Use a Clark-like clustering
method, where not all word-senses will be categorized.

Current work is to implement step 1) above in [word_senser.py](src/word-senser.py), 
using the following steps:
- First pass
   3) Store sentences in corpus in order 
   1) Calculate and store embeddings for each word in the corpus in a
   nested list [[e1, e2, ..] [e1, e2, ..] .. [e1, e2, ..]]
   2) Store a dictionary with the sentence and word position for each 
   word in the corpus
- Second pass
   1) For each word in vocabulary with more than threshold frequency, 
   gather all its embeddings.
   2) Cluster such embeddings to obtain word senses.
   
*******************

There's a working implementation of WSD in word_senser.py

After experimenting with 
[senseval2_lexical_sample_train](../UFSAC/corpus/ufsac-public-2.1/senseval2_lexical_sample_test.xml), 
I notice that memory consumption is quite large using
the concatenation of the last 4 hidden states.
In order to keep testing in my laptop, I change to using ~~only the 4th to last
hidden layer.~~
~~TODO: Switch to the average of the last 4 layers, since that is the second
best result obtained by Devlin et al. (2019).~~
the arithmetic average of the last 4 layers.

I confirm that using KMeans, which requires a fixed number of clusters,
spreads similar meanings to different clusters.
However, the division is not bad for the actual ambiguous terms, 
and "saw" and "bank" are properly disambiguated in this case.
~~TODO: However, it's a good idea to try some agglomerative methods.~~

~~TODO: convert variables to class variables~~

*******************
Decided to try sklearn clustering algorithms which don't necessarily cluster
all words.

Tried DBSCAN with a few different parameters: it's clear that the param
that decides the minimum distance for a cluster is crucial, so then
decided to try OPTICS, which allows to use a range for that value, to
see which gives more reasonable clusters.

One thing noted with DBSCAN is that the clustering is not bad.
When the appropriate parameters are chosen, words are disambiguated in
clusters that sometimes portray the same meaning, but one or two clusters
give a very distinct meaning.
Very little overlap between meanings was observed in the same clusters.

Parameters decided for OPTICS method:
- min_samples: 0.1 (10% of all occurrences of a word are the min to define 
a cluster)
- metric: cosine
 
 ******
 Decided to output WSD results in appropriate format to evaluate with AdaGram's
 evaluator against GOLD standard.
 This will help guide the parameters to use for clustering.
 Tune parameters with training data, evaluate against test data.
 
~~TODO: In reference corpus, words disambiguated in some sentences are 
not considered ambiguous in other sentences.
 Will attempt to export to disamb file only those that are disambiguated in
 key corpus.~~ DONE
 
 ***********
 There is only one instance of `colorless` in the 
 `senseval2_lexical_train.xml` corpus, which in bert-sense is counted together
 with all the `coloUrless` occurrences through use of the lemmatization,
 but in my code it gets distinguished, so it causes problem with clustering that
 it occurs only once.
 I guess I should always use threshold > 1
 
 TODO (efficiency improvement, non-braking): Convert ambiguous_gold to a set
 
 **************
 
 Modified AdaGram's [test-all.py](src/test-all.py) to evaluate the
 disambiguation results.
 TODO: Fix problem causing division by zero, and evaluate results in nova.
 
 
 ************
 After a new discussion with Ben, he came up with another way of creating
 word categories that takes care of disambiguation in the process!
 
 The idea is as follows:
 - For a series of sentences, mask one (random?) word in each.
 - Using BERT, get the logit prediction for the masked word in a sentence.
 This "embedding" for the predicted word will be similar to those of
 other sentences where the mask accomplishes a similar semantic and 
 syntactic function.
 Even if semantics are involved, the syntactic requirement will play
 a very important part, as BERT predicts grammatical sentences.
 - Cluster the vector predictions for all the sentences.
 - Form a word category by adding together the top predictions for
 all the vectors in a given cluster.
 
 Note that this process disambiguates word senses, as the same word can 
 have a high probability of occurring in predictions for different clusters.
 
 First try with just a few sentences, clustered with KMeans, in 
 [wordcat_bert.ipynb](notebooks/wordcat_bert.ipynb).
 Masked words are only adjectives, personal nouns, location nouns.
 Obtained word categories look quite decent, and of course the granularity
 depends on the number of clusters used:
 
 Sentences:
- The _ cat ate the mouse.
- She was wearing a lovely _ dress last night.
- He was receiving quite a _ salary.
- He also bought a _ sofa for his new apartment.
- I was born and grew up in _.
- The _ metropolitan area added more than a million people in the past decade.
- Bike races are held around the _ and farmlands.
- My racist _ called me last night.
- A device is considered to be available if it is not being used by another _.
 
 Clusters with k = 3 (3 clusters)
 - Category 0:
little, modest, evening, handsome, new, black, generous, luxury, great, silver, blue, white, gray, yellow, old, green, brown, comfortable, respectable, small, low, fine, giant, good, decent, leather, silk, substantial, pink, red, dead, mother, steady, big, wild, luxurious, nice, high, purple, wedding, large, fat, considerable, cheshire

 - Category 1:
indianapolis, village, washington, toronto, lake, london, mountains, city, atlanta, california, woods, austin, cleveland, mexico, brooklyn, forest, hills, gardens, philadelphia, countryside, denver, chicago, minneapolis, florida, seattle, portland, farms, parks, dallas, germany, towns, detroit, france, park, lakes, louisville, texas, england, pittsburgh, villages, forests, houston, town, canada, fields

 - Category 2:
cousin, neighbors, friend, entity, dad, boyfriend, customer, roommate, friends, application, person, organization, father, party, neighbor, wife, company, partner, mother, girlfriend, provider, brother, user, device, uncle, boss, husband
 
 Clusters with k = 4
 
 - Category 0:
modest, cousin, handsome, generous, neighbors, friend, dad, boyfriend, roommate, friends, comfortable, respectable, low, small, fine, decent, good, father, boss, substantial, neighbor, wife, partner, steady, mother, girlfriend, brother, nice, high, uncle, large, considerable, husband

- Category 1:
indianapolis, village, washington, toronto, lake, london, mountains, city, atlanta, california, woods, austin, cleveland, mexico, brooklyn, forest, hills, gardens, philadelphia, countryside, denver, chicago, minneapolis, florida, seattle, portland, farms, parks, dallas, germany, towns, detroit, france, park, lakes, louisville, texas, england, pittsburgh, villages, forests, houston, town, canada, fields

- Category 2:
little, evening, new, black, luxury, great, silver, blue, white, gray, yellow, old, green, brown, comfortable, small, giant, leather, silk, pink, red, dead, mother, big, wild, luxurious, purple, wedding, large, fat, cheshire

- Category 3:
user, party, application, device, entity, company, person, customer, provider, organization
 
 Clusters with k = 5
 
 - Category 0:
indianapolis, village, washington, toronto, lake, london, mountains, city, atlanta, california, woods, austin, cleveland, mexico, brooklyn, forest, hills, gardens, philadelphia, countryside, denver, chicago, minneapolis, florida, seattle, portland, farms, parks, dallas, germany, towns, detroit, france, park, lakes, louisville, texas, england, pittsburgh, villages, forests, houston, town, canada, fields

- Category 1:
little, evening, new, black, luxury, great, silver, blue, white, gray, yellow, old, green, brown, comfortable, small, giant, leather, silk, pink, red, dead, mother, big, wild, luxurious, purple, wedding, large, fat, cheshire

- Category 2:
user, party, application, device, entity, company, person, customer, provider, organization

- Category 3:
nice, modest, handsome, decent, high, comfortable, steady, substantial, generous, respectable, low, small, fine, large, considerable, good

- Category 4:
father, cousin, friends, boss, neighbors, husband, friend, dad, uncle, neighbor, wife, boyfriend, partner, mother, girlfriend, brother, roommate
 
 Clusters with k = 6
 
 Category 0:
nice, modest, handsome, decent, high, comfortable, steady, substantial, generous, respectable, low, small, fine, large, considerable, good

Category 1:
little, evening, new, black, luxury, great, silver, blue, white, gray, yellow, old, green, brown, comfortable, small, giant, leather, silk, pink, red, dead, mother, big, wild, luxurious, purple, wedding, large, fat, cheshire

Category 2:
father, cousin, friends, boss, neighbors, husband, friend, dad, uncle, neighbor, wife, boyfriend, partner, mother, girlfriend, brother, roommate

Category 3:
indianapolis, washington, toronto, london, atlanta, california, austin, cleveland, mexico, brooklyn, philadelphia, denver, chicago, minneapolis, florida, seattle, portland, dallas, germany, detroit, france, louisville, texas, england, pittsburgh, houston, canada

Category 4:
village, forest, hills, forests, gardens, countryside, towns, parks, lake, park, lakes, town, mountains, city, farms, villages, fields, woods

Category 5:
user, party, application, device, entity, company, person, customer, provider, organization


In this example, 6 clusters seems like the best result with the above sentences.
 ***********
 
 TODO:
 ~~- Organize code in functions~~
 - Try clustering with DBSCAN/OPTICS, which leave some words unclustered
 - Automate masking every word in the sentence
 - Add some voting system to decide which words go in a cluster? Currently
 all "high" words in every member of a cluster go into it.
 - Handle sub-words in sentences
 ~~- Try with large bert model~~
 
 ********
 Also tried masking every word in the above sentences and clustering that way.
 Results can be found in [wordcat_funcs.ipynb](notebooks/wordcat_funcs.ipynb).
 The categories are not as crisp and in the former, simpler case... they
 contain some noisy results.
 
 ********
 ## Feb 2020

After reporting the clustering results above, Ben pointed out that the idea
he had in mind is different.
I was clustering sentences, and obtaining the word categories from them in
a way that circumvents WSD.
However, as Ben noted, this won't work in general, as you can have 
syntactically different predictions from the logit vectors:
```
Sentence:
The fat cat [MASK] the mouse.
Ordered top predicted tokens: ['.', 'and', ',', 'or', ';']
Ordered top predicted values: [0.5628975  0.32685623 0.05045042 0.00994903 0.0039914 ]
```

Instead, Ben proposes to fill the sentence-word matrix with the probabilities
of the given masked sentence, if the mask is substituted by the given word.
`P(S_i|W)`: Probability of sentence `S_i`, which has one masked position, 
if word W is placed instead of the mask.
That is: `Ai_j = P(S_i|W=j)`
This approach will require several more evaluations than the one I tried
above (a factor of around `V*avg_l` more evaluations, where V is the size of the
vocabulary, and `avg_l` is the average sentence length).
However, it seems intuitive that his approach should work better.
This approach would require a separate sense disambiguation, though... perhaps
the one implemented in [word_senser.py](src/word_senser.py).

********
While reviewing code in [Bert_as_LM.ipynb](notebooks/Bert_as_LM.ipynb), I
notice that the normalized sentence probability is taking the sentence length
as the number of sub-words.
For most words, the sub-words have normally quite high probability, so this
may be biasing probabilities in favor of sentences with a high ratio of 
sub-words.
TODO: Maybe I should divide by the length of the sentence in words, not counting
sub-words?

Also, the probabilities of subworded words is much higher than regular
words, because I'm masking one subword at a time, so the other subwords
can have high probabilities because they fit well with the surrounding
subwords, even if not too much with the rest of the sentence.
Example:
The sentence `This is a macrame` should have quite a low probability, but
because of sub-words, it's actually quite high:
```
Processing sentence: ['[CLS]', 'this', 'is', 'a', 'mac', '##ram', '##e', '.', '[SEP]']
['[CLS]', '[MASK]', 'is', 'a', 'mac', '##ram', '##e', '.', '[SEP]']
Word: this 	 Prob: 0.06995750218629837
Ordered top predicted tokens: ['it', 'this', 'there', 'he', 'that']
Ordered top predicted values: [0.8618048  0.0699575  0.02746578 0.00282901 0.00164153]
['[CLS]', 'this', '[MASK]', 'a', 'mac', '##ram', '##e', '.', '[SEP]']
Word: is 	 Prob: 0.8059344291687012
Ordered top predicted tokens: ['is', 'was', 'has', 'resembles', 'includes']
Ordered top predicted values: [0.8059344  0.14677647 0.00251259 0.00214571 0.00181202]
['[CLS]', 'this', 'is', '[MASK]', 'mac', '##ram', '##e', '.', '[SEP]']
Word: a 	 Prob: 0.3107656240463257
Ordered top predicted tokens: ['a', 'the', 'called', 'not', 'another']
Ordered top predicted values: [0.31076562 0.24131534 0.1898963  0.01890805 0.01177635]
['[CLS]', 'this', 'is', 'a', '[MASK]', '##ram', '##e', '.', '[SEP]']
Word: mac 	 Prob: 0.4050573706626892
Ordered top predicted tokens: ['mac', 'ce', 'side', 'cup', 'semi']
Ordered top predicted values: [0.40505737 0.0790267  0.05402635 0.0435847  0.03299322]
['[CLS]', 'this', 'is', 'a', 'mac', '[MASK]', '##e', '.', '[SEP]']
Word: ##ram 	 Prob: 0.2340756058692932
Ordered top predicted tokens: ['##ram', '##ula', '##out', '##abe', '##are']
Ordered top predicted values: [0.2340756  0.13345419 0.10716671 0.05448193 0.03716443]
['[CLS]', 'this', 'is', 'a', 'mac', '##ram', '[MASK]', '.', '[SEP]']
Word: ##e 	 Prob: 0.7259987592697144
Ordered top predicted tokens: ['##e', '##id', '##i', 'plant', 'virus']
Ordered top predicted values: [0.72599876 0.0952542  0.03262086 0.01331557 0.00876465]
['[CLS]', 'this', 'is', 'a', 'mac', '##ram', '##e', '[MASK]', '[SEP]']
Word: . 	 Prob: 0.9112837910652161
Ordered top predicted tokens: ['.', ';', '!', '?', '|']
Ordered top predicted values: [9.1128379e-01 7.5437672e-02 8.2414346e-03 4.5370557e-03 3.7746594e-04]

Normalized sentence prob: log(P(sentence)) / sent_length: -0.9733260146209172
```
TODO: Handle subwords differently

**************
File [word_categorizer.py](src/word_categorizer.py) contains the current
attempt with Ben's approach discussed above.

Some notes:
- Start with simple English vocab (some files in [vocabularies](vocabularies))
- Start with small number of sentences.
- Probably should use sparse vectors and assign very low sentence probs
 (below some threshold) as zeros: ~~TODO~~ Done, but doesn't seem to make
 a lot of difference.
 ~~Need to make sure that clustering can handle sparse vectors, or else~~
 - ~~Should~~ Could implement Clark-style clustering algo.
 - Currently, I don't think it's worth to reuse BERT evaluations for 
 different words... I think saved processing time is small compared to
 added algorithm complexity
 - How to deal with sub-words? Both for probabilities and for substitution
 
 **********
 Need an evaluation measure for the word categories.
 Following a [list of words by POS](https://www.english-grammar-revolution.com/word-lists.html),
 I create a gold standard to evaluate the obtained categories.
 Labeled words per category are in [POS](vocabularies/POS) directory.
 The file [POS_unambiguous.vocab](vocabularies/POS_unambiguous.vocab)
 contains these words, eliminating all words appearing in more than one
 category, to avoid dealing with ambiguity for now.
 
 ***********
 Actually, ambiguity is still present in the above, as some words appearing
 in the lists can be used as different POS, and they are clustered only
 in a single category, i.e. `match` and `love` get normally clustered together with
 nouns, instead of verbs.
 We should handle ambiguity to more properly evaluate the clustering.
 
 Clustering must be done carefully.
 The standard k-means does quite a decent job for a number of clusters 
 similar to the number of expected POS in the gold standard.
 On the other hand, DBSCAN is very sensitive to the `eps` parameter
 which decides the density of the clusters.
 If `eps` changes a bit outside of the optimal value, all words get 
 clustered in the same cluster, or not clustered at all.
 DBSCAN doesn't seem so robust in this sense.
 Keep in mind to implement Clark-clustering?
 
 I wonder if this is also related to having "a few" features (less than 190
 for the current experiments: 19 sentences with at most 10 masks).
 
******************

Using the OPTICS algorithm with default values gives a different type of
clustering: a lot of words are left unclustered (cluster -1), but
the formed clusters are quite good, if small.
Example:
```
Cluster #-1:
[airline, am, and, anyone, around, as, asian, at, audience, away, baby, badly, band, bed, between, brutally, building, but, by, christian, clock, country, cut, daily, did, do, drive, during, eat, entire, extremely, family, few, final, flowers, fly, french, from, go, grimly, grow, has, have, history, house, it, little, make, man, may, most, mountain, movie, music, never, none, ocean, off, of, on, or, organized, phone, play, pregnant, pretty, quite, sleep, socks, something, soon, state, stop, sunlight, swim, than, themselves, though, too, to, train, up, upstairs, urgently, very, village, walk, weekly, well, what, which, while, will, with, work, write, yesterday, yet, you, ]
Cluster #0:
[above, behind, below, beside, under, ]
Cluster #1:
[across, onto, through, ]
Cluster #2:
[everywhere, here, somewhere, there, ]
Cluster #3:
[carefully, eagerly, happily, quickly, quietly, ]
Cluster #4:
[australia, eyeglasses, kilimanjaro, lazily, santiago, ]
Cluster #5:
[banana, milk, rain, rice, snow, violin, ]
Cluster #6:
[bird, cat, dog, ]
Cluster #7:
[book, match, photograph, ]
Cluster #8:
[happiness, love, wealth, ]
Cluster #9:
[myself, yourself, yours, ]
Cluster #10:
[bad, clean, cold, dry, funny, good, hot, sharp, ]
Cluster #11:
[him, me, them, ]
Cluster #12:
[become, been, be, seem, ]
Cluster #13:
[because, if, once, until, when, where, ]
Cluster #14:
[whoever, whom, who, ]
Cluster #15:
[i, she, they, we, ]
Cluster #16:
[can, must, should, ]
Cluster #17:
[some, these, this, ]
Cluster #18:
[her, his, my, ]
Cluster #19:
[are, is, was, ]
```
********
## Feb 12, 2020
See [mini-report](results_feb11.md) shared this day on slack.

**********
## Feb 13, 2020
After sharing [an explanation](handle_subwords.md) of the sub-word problem
in slack, Ben pointed out that the way I calculate sentence probability
is flawed from a probabilistic point of view.
I was aware of this, as it had been noted on the web 
[here](https://github.com/google-research/bert/issues/139) and
[here](https://github.com/google-research/bert/issues/35), but I had taken
the approach because it was implemented similarly in 
[here](https://github.com/xu-song/bert-as-language-model) and
[here](https://github.com/huggingface/transformers/issues/37).

My approach is clearly causing trouble with sub-words, and as Ben noted,
also with common phrases like "quid pro quo".
He suggests to calculate the probabilities right, which would solve the
sub-word problem (with regards to probability estimates, I still need
to make sure I substitute whole words and not just tokens when comparing
sentence probabilities for different words):
```
P_forward(he answered quickly .) = P(he) * P(he answered | he) * P( he answered quickly | he answered) * P(he answered quickly . | he answered quickly) 
```

Since this would ignore BERT's bidirectional capabilities, Ben proposes to also
calculate a similar probability but going backwards:
```
P_backwards(he answered quickly .) = P(.) * P(quickly .| .) * P(answered quickly .|quickly .) * P(he answered quickly .|answered quickly .) 
```
and geometric-average the two:
```
P(he answered quickly .) = sqrt(P_forward * P_backwards)
```

******
## Feb 14, 2020

Because joint probability is defined as:
```
P(A, B, C) = P(A) * P(B|A) * P(C|A, B)
```
I think a term like `P(he answered|he)` actually means
`P(answered|he)`

The question remains on how to actually interpret terms like
`P(he answered|he)` while working with BERT.
I interpreted it two ways:

The first way is to grow the sentence gradually, and to calculate terms
like
```
P(w_1=answered|w_0=he)
```
where the sentence fed into BERT would be:
```
['[CLS]', 'he', '[MASK]', '[SEP]']
```
and I would get the probability of `P([MASK]=answered)` from applying
softmax to the output of the last layer in BERT.

This approach, which grows the sentence gradually, was attempted in
[a notebook](notebooks/Bert_as_LM_unidirectional-fail.ipynb), but seems
failed to me.
Since the sentences fed into BERT are sub-parts of the original one
and would be quite different it, the probabilities are really off.

Alternatively, I think it makes more sense with BERT to feed it a
sentence with the right length, but masking all words that are not
involved in the current probability estimation.
I.e. to calculate `P(he answered|he)` in the sentence `He answered quickly.`,
I would feed the sentence 
```
['[CLS]', 'he', '[MASK]', '[MASK]', '[MASK]', '[SEP]']
```
and again I take the probability of `P([MASK]=answered)` from applying
softmax to the output of the last layer in BERT.
This is the approach taken in 
[this notebook](notebooks/Bert_as_LM_unidirectional.ipynb), which
seems to make a lot more sense probabilistically.

*******
## Feb 15, 2020
Also, the probability of a sentence as calculated above would grow with the 
sentence length. 
Hence, I decide to use some normalization.
Previously, as in [this notebook](notebooks/Bert_as_LM.ipynb) I was actually
calculating the logarithm of the sentence probability, divided by the length
of the sentence, i.e. `log(P(S))/len(S)`.
This is equivalent to the logarithm of the geometric average of the
probabilities of each prediction.
Instead of representing the vectors in log space, I decided to keep
the geometric average of the components of the probability estimation:
```
P_forward(he ran .) = (P(he) * P(he ran| he) * P(he ran .|he ran)) ^ (1/3)
```
[This notebook](notebooks/Bert_as_LM_unidirectional.ipynb) contains calculations
performed this way.
It doesn't make as much sense as the previous attempt, because sentences like
`The steak was eaten by the man`, and `The steak ate the man` are not too dissimilar.
However, I will try plugging it to the word category formation code, to check if the
results make better sense than before.

*****
## Feb 17, 2020

