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
first updated [Bert_Model.py](../src/BERT_Model.py) to use the latest huggingface transformers models 
(github.com/huggingface/transformers) and to be able to run without CUDA 
(fixed a bug).
The obtained results were not exactly the same as their paper, but quite close.
I blame the change of transformers versions for the slight difference.

Then, I wanted to test my idea of just getting embeddings for every word
in the corpus, then cluster them and hope that word categories will appear
as a result.
This idea is coming from the previous use of word2vec and AdaGram vectors
for word categorization. 
The code for this attempt is at [all_word_senses.py](../src/all_word_senses.py).
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

Current work is to implement step 1) above in [word_senser.py](../src/word_senser.py), 
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
[senseval2_lexical_sample_train](../../UFSAC/corpus/ufsac-public-2.1/senseval2_lexical_sample_test.xml), 
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
 
 Modified AdaGram's [test-all.py](../src/test-all.py) to evaluate the
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
 [wordcat_bert.ipynb](../notebooks/wordcat_bert.ipynb).
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
 Results can be found in [wordcat_funcs.ipynb](../notebooks/wordcat_funcs.ipynb).
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
the one implemented in [word_senser.py](../src/word_senser.py).

********
While reviewing code in [Bert_as_LM.ipynb](../notebooks/Bert_as_LM.ipynb), I
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
File [word_categorizer.py](../src/word_categorizer.py) contains the current
attempt with Ben's approach discussed above.

Some notes:
- Start with simple English vocab (some files in [vocabularies](../vocabularies))
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
 Labeled words per category are in [POS](../vocabularies/POS) directory.
 The file [POS_unambiguous.vocab](../vocabularies/POS_unambiguous.vocab)
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
[a notebook](../notebooks/Bert_as_LM_unidirectional-fail.ipynb), but seems
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
[this notebook](../notebooks/Bert_as_LM_unidirectional.ipynb), which
seems to make a lot more sense probabilistically.

*******
## Feb 15, 2020
Also, the probability of a sentence as calculated above would grow with the 
sentence length. 
Hence, I decide to use some normalization.
Previously, as in [this notebook](../notebooks/Bert_as_LM.ipynb) I was actually
calculating the logarithm of the sentence probability, divided by the length
of the sentence, i.e. `log(P(S))/len(S)`.
This is equivalent to the logarithm of the geometric average of the
probabilities of each prediction.
Instead of representing the vectors in log space, I decided to keep
the geometric average of the components of the probability estimation:
```
P_forward(he ran .) = (P(he) * P(he ran| he) * P(he ran .|he ran)) ^ (1/3)
```
[This notebook](../notebooks/Bert_as_LM_unidirectional.ipynb) contains calculations
performed this way.
It doesn't make as much sense as the previous attempt, because sentences like
`The steak was eaten by the man`, and `The steak ate the man` are not too dissimilar.
However, I will try plugging it to the word category formation code, to check if the
results make better sense than before.

***************
## Feb 17, 2020
Implemented changes to make unidirectional sentence probability calculation.
Ran OPTICS experiment; results are as good as previous runs.
Also got distracted by meeting + debugging colleague's code 
for networks project.

***************
## Feb 18, 2020

Implemented sub-word handling while replacing words in a given sentence,
to fill the word-sentence matrix used for word category formation.
This is necessary, regardless of the sentence probability estimation.
Ran experiment.

Also, went back to WSD, as this is the next step to add into the pipeline.

********
## Feb 21, 2020
In the past days, I reviewed the code and checked performance of WSD, which
is the stage I had stopped at last time I worked on it.
Performance is evaluated against the gold standards in the senseval/semeval
files. The problem with those is that:
1) They are quite granular in the senses they provide.
2) There are too few examples to work with

Hence, sklearn clustering algorithms DBSCAN and OPTICS are struggling to 
disambiguate. K-means does a decent job by eye, but bad against senseval
for the above-mentioned reasons.

I started connecting the WSD code output with the word_categorizer, as
the latter needs input from the former. 
I think I'll store the word-sense centroids in a pickle file, which
can later be read by the word categorizer.

Later: Implemented export of cluster centroids.

****************
## Feb 22, 2020
Implemented word senser from plain text file, instead of xml file with
word-sense disambiguation.
This is needed for the real case where we're processing plain, unlabeled
text.

Placed XML senser in [word_senser_XML.py](../src/word_senser_XML.py)

************
## Feb 23, 2020
Created a small [WSD corpus](../sentences/smallWSD_corpus.txt) and made ran experiments with the latest
word_senser.py and different clusterings.
OPTICS is not working at all in these cases, not sure why. It gives either
0 or 1 clusters, even varying the max_eps distance parameter.
KMeans clustering does a decent job disambiguating words in the new corpus.

Next: Modify word_categorizer to take disambiguated sense and generate a
different embedding per sense.

~~TODO: Make sure only disambiguated words are exported in centroids~~

**************
## Feb 24, 2020

Word senser only exports embeddings for ambiguous words now.

Implemented the use of WSD in the word categorizer.
Running experiments overnight.

**********
## Feb 25, 2020

Experiment with OPTICS gives good clusters, but several words are left 
unclustered.
~~Trying with KMeans gives out of memory error.~~ FIXED.

A word on metrics: SKlearn KMeans implementation cannot use "cosine" distance.
I'm using cosine distance in OPTICS.

Checking ways to reduce memory use after the matrix has been built:
- release both BERT models: ~~TODO.~~ DONE
- don't use sparse matrix. I think if there are no zero entries (we aren't
using zero entries now) this consumes more memory? Maybe wait to change this.
~~TODO.~~ DONE
- once matrix is built, no need to hold centroids in memory. ~~TODO.~~ DONE

Seems like memory problem comes from faulty matrix storage, not from
above modules.
However, good idea to not even load BERT modules and centroids when loading
matrix from pickle: less time and memory consumption, for later larger runs.
FIXED

Word category formation runs take a very long time.
SmallWSD run (16 sentences) takes about 12 hours in my laptop.
19sentences.txt takes about 24 hours in nova.

Finding ways to improve processing time.
And results are still not properly tuned... so going through pipeline
slowly with debugger, to make sure everything does what I expect, and
check for speed improvements

Looking at the embeddings for word_senser, Ben is right that they do not
necessarily work for WSD. They seem different, even for almost identical
uses of the word, even in same sentence.
It's curious that KMeans still can disambiguate them decently.

************
## Feb 26, 2020
Wrote [report](report_feb26.md) about current status:
```
I added handling of sub-words, modified the way sentence probabilities
are calculated, and integrated disambiguation into the pipeline.
After these changes, the clusters obtained are still good, but several
words are still uncategorized.
```

***********
## Feb 27, 2020
Made some tests on current processing time.
I had the impression that past runs were not scaling properly.
After a few timing experiments, I noticed I was misestimating.
Actually, scaling is as expected: smallWSD corpus with 15 sentences and
10 masks should last around 9 hours, as observed (more or less).

Potential efficiency improvements:
Calculate sentence probabilities for a sentence more cleverly:
many of the BERT evaluations are repeated in the same sentence.
I.e. when evaluating the forward and backward probability, many of the
passes do not involve the current word being evaluated, so they can be 
reused.
E.g. when evaluating `This is a ______ sentence .`, for each vocabulary
word I currently evaluate `P(w_0 = This| MASK MASK MASK MASK MASK MASK)`.
This (and many others) could immediately be reused by any of the vocabulary
word evaluations.
I think I could cut evaluation at least by half with this.

TODO: Find a clever way to calculate sentence probabilities for all the
vocabulary.

Also, other sentences could benefit from some of these evaluations if they
were saved.
Ben suggested using [tries]()

*******
## Mar 18, 2020

Started redesign of WSD to use BERT sentence probabilities instead of 
final layers embeddings.
Firs test is running overnight, trying to WSD smallWSD corpus.

**********
## Mar 19, 2020

On slack, Ben wrote today:
*******
@asuarez regarding the scalable version of the NeurOracular ULL approach... it seems there are two biggish things to work out... (i.e. things going beyond what we did in the POC work)

10:07
One is how to get a lot of sentence probabilities calculated in a feasible amount of compute resources.  My suggestion there was to build up a trie of word-sequence probabilities, filling it up incrementally as one makes each masked-word or masked-sentence prediction using the language model (using all the probabilistic predictions the language model makes for the masked word/sentence, not just the top prediction).   This should be straightforward-ish but will require attention to implementation details esp. making sure the trie implementation can handle huge tries, and ideally can handle the same trie being updated concurrently via multiple processors (which are each parsing separate portions of the corpus).
*******

My response:

@ben, I agree these are two points to work on.

Describing a bit the situation for @andre and others, about saving word-sequence probabilities, and also want to understand what you suggest to store:

With the current way to calculate the probability of sentence "Not a real sentence", we need to ask Bert's Masked Language Model (MLM) prediction the following probabilities:

```
FORWARD:
a) P(M1 = Not           |M1 M2 M3 M4)
b) P(M2 = a             |Not M2 M3 M4)
c) P(M3 = real          |Not a M3 M4)
d) P(M4 = sentence      |Not a real M4)
BACKWARD:
e) P(M4 = sentence      |M1 M2 M3 M4)
f) P(M3 = real          |M1 M2 M3 sentence)
g) P(M2 = a             |M1 M2 real sentence)
h) P(M1 = Not           |M1 a real sentence)
```
In some of the processes, e.g. WSD, I need to replace a word in the sentence with every other word in the vocabulary. For example, I would fill "Not ___ real sentence", and thus evaluate things like "Not dog real sentence", "Not quickly real sentence", etc.

We can see that from the probabilities above, a), e) and f) are completely reusable as they are, for any word that fills the blank. Also, a) and e) can be obtained from the same MLM evaluation of Bert. Still more, b) and g) can also be reused for every word filling the blank, if we save all word predictions that BERT makes for M2 in them. So, only c), d) and h) need to be re-evaluated for every different word filling the blank.

I am in the process of redesigning the code for such calculations to reduce approx half the computations, for sentences where we have to fill the blank with all vocabulary words.
As this process is needed for WSD, I save a matrix with all sentences-with-blank vs vocabulary scores, which can be reused for word-category formation without the need to reevaluate anything in BERT.

Now, the remaining evaluations c), d) and h), when evaluating fill-the-blank sentences, won't be very repeatable. E.g., for h) we will be evaluating things like

h) P(M1 = Not           |M1 dog real sentence)

or

h) P(M1 = Not           |M1 quickly real sentence)

which I don't feel it's even worth storing in a trie. Or is it?

I guess storing them may potentially be useful for random generated sentences.
@ben this is what you have in mind to store in the trie, right? 
Not only a given sentence probability as P(Not a real sentence).

******************
On the results of the first test, there was a memory error when clustering.
Checking what went wrong, there shouldn't be a memory error with the
smallWSD corpus.

***********
Added tokenizer-only class, to be used in wordsenser.

******************
## 20 Mar, 2020

Slack exchange. Ben says:

500 million sequences would be nice, figuring average sequence length of 20

This is a rough estimate based on 40GB text file training data for GPT2, figuring 200M words or so per GB of text-file

************
My reply:

@ben for a GPT2-sized training corpus, you're right that there
will be around 500M sentences (or word sequences), but to estimate the
probability of a sentence of length N, we need 2*N - 1 BERT predictions.
 
For WSD and word-category formation, we then need to replace one word 
of the sentence with a different one (fill in the blank). We can
reuse half of those predictions, but still need to calculate around N of 
them for each word in the vocabulary of interest for the task.
Then, we replace the next word in the sentence with a blank and fill it
with all words in the vocabulary of interest. 
Again, we can reuse about half of the predictions, but still need to 
calculate some new ones.
And so on for each position in the sentence.

Now, for grammar induction, many of the produced sentences are going to
be ungrammatical, so probably not pre-stored in the trie from the sequences
calculated in previous steps; meaning we also need to ask BERT for around
N new predictions for them.

So, I think if we were to use such large corpora, we would need way more
than 500M stored values.
Of course, we are probably going to use smaller corpora, and storing them
on the fly as we calculate them (if not in the trie already) makes sense.

@andre, it may also be relevant that the data that will be stored in the 
trie are arrays with one floating point value per vocabulary word (in the 
case of BERT's large model, that's around 30k values in every BERT 
prediction).

**************
## 23 March, 2020
In the past days, I changed the WSD to use sentence probabilities instead
of BERT's final layers.
Algorithm is working, and test is running to see the quality of the produced
word senses.

Also, redesigned WSD to reuse around half of the sentence probabilities
used in the process, so it runs faster.
In my laptop, running time decreased to less than half, which is what I
expected.
Running a longer experiment now in `nova` to see speedup.
I made sure during debugging that the sentence probabilities obtained are
the same as for a single sentence in the jupyter notebook, so I'm quite
confident about it both for sentences with and without sub-words :)

@senna asked on slack to give a high level description of the algorithm
where I would be using the tries.
***************
@senna:
Yes I think we need a tailor-made one. Specially because of this comment from Ben: "and ideally can handle the same trie being updated concurrently via multiple processors (which are each parsing separate portions of the corpus)."  But also because your problem have some details that would be better explored by using a specialized data structure (for example the "mask position" integer)

I'm currently working in a possible design for it.
Do you have a higher level description of the algorithm?
************
@glicerico:
Sure. There is only one part where we would need to write or read from the trie, and that is for sentence probability evaluation. Each sentence probability evaluation requires a number of BERT predictions, which is what we will be storing in the trie.

Not so high level, but this is how we calculate a sentence probability P(S):
```
P(S) = (P_f * P_b) ^ (1/2), where
P_f = P(w_0) * P(w_1|w_0) * P(w_2|w_0, w_1) * ... * P(w_N)
P_b = P(w_N-1|w_N) * P(w_N-2|w_N-1, w_N) * ... * P(w_0|w_1, w_2, ... ,w_N)
```
and each factor in P_f and P_b requires a BERT prediction (or trie lookup).

There's two main parts where we need to use sentence probabilities in the process.

First is word sense disambiguation (WSD), where we build word-instance vectors where each vector component is a sentence probability replacing the word-instance of interest with all other words in the vocabulary. Then we cluster those vectors to get word-senses.

Second part is grammar induction, where we randomly generate sentences that follow a given explicit grammar and compare their probability against sentences generated from a distortion of that grammar. This way, we evaluate if a grammar is close to the grammar implicit in BERT, and it's an iterative process to approximate the grammar in BERT.

We submitted a paper explaining with more detail, let me know if that would be useful.

****************
## Mar 26, 2020

WSD results from the oracle approach are not as good as they were before,
using final layer embeddings.
I've been trying to find out the reason for that.

Current normalization using sentence length shows high variation:
Grammatical sentences with the same length can have large sentence
probability differences, according to the training data for BERT.
E.g.
```
Processing sentence: ['[CLS]', 'the', 'fat', 'cat', 'ate', 'the', 'last', 'mouse', 'quickly', '.', '[SEP]']
1.5699080230392615e-23

Processing sentence: ['[CLS]', 'there', 'are', 'many', 'health', 'risks', 'associated', 'with', 'fat', '.', '[SEP]']
8.689618845171052e-20

Processing sentence: ['[CLS]', 'they', 'will', 'fly', 'out', 'of', 'santiago', 'tomorrow', 'morning', '.', '[SEP]']
3.506387431159266e-22

Processing sentence: ['[CLS]', 'she', 'bought', 'the', 'last', 'microsoft', 'mouse', 'for', 'santiago', '.', '[SEP]']
2.933483879831623e-28
```

For WSD, all fill-the-blank sentences share context and so the
features should be comparable to each other.
For comparison to other word-instances in different sentences, however, we
need to normalize each word-instance embedding to unit length, so that the
pair-wise distances are not dominated by a sentence which has an inherently 
larger probability.

Implemented normalization.

Still, the clusters are not good.
To help exploration, I implemented a visualizer for the word-instance
embeddings.
It shows projections of the embeddings using the two main PCA components, and
also the t-SNE algorithm.
The K-Means obtained clusters correspond fine with the PCA projections, but not
as well for the t-SNE projections.
However, the t-SNE projections also don's show a correct division of word
senses as we would expect them.

Going through some of the clearer polysemous word examples like "fat" and "fly",
I notice that indeed the [smallWSD corpus](../sentences/smallWSD_corpus.txt)
doesn't provide a good context for disambiguation.
The hope is that a more extended corpus will work better for this.
However, this may be a false hope.

I want to build a corpus which, even with only a couple ambiguous words, I can
clearly distinguish them and cluster them correctly.
UPDATE: Created a [corpus](../sentences/fat_saw_corpus.txt) with two clearly
ambiguous words: "fat" (noun and adj) and "saw" (verb and noun).
Will run it through WSD.

Rethinking about sentence-length normalization... I wonder if it makes any
sense in the WSD context.
If I am going to normalize to unit length vector later, perhaps I can ignore it.
Trying that in Sergey's cluster to compare results.

****************
## May 1, 2020: Fierce Mayday to you all!

Coming back to this after some weeks focused in agent-based simulations.

Updates which were not logged before:

- The [fat_saw corpus](../sentences/fat_saw_corpus.txt) is processed okay with
KMeans, not with the OPTICS/DBSCAN algos.
- normalizing to unit-length seems to work fine. 
Will use this instead of normalizing by sentence length.
- Tried using BERT model only in forward mode, not backward.
This gave results as good as using both ways... wonder if I could just get rid
of backwards calculation.

More recent developments...
- Implemented spherical clustering methods from 
[spherecluster](https://github.com/jasonlaska/spherecluster); reported and fixed
a bug in this repo.
All three tested methods (SphericalKMeans, 2 mixture of von Mises-Fisher
 distributions) provide exactly the same results as KMeans, and match well with
 PCA visualization of embeddings.
 
 ## May 2, 2020
 Tried forward-only WSD with bot fat_saw corpus and smallWSD.
 Using KMeans for comparison, both give similar (not perfect) results 
 as bidirectional
 (forward and backwards) sentence probabilities.
 Makes me think maybe can use this simplified calculation.
 TODO: Implement forward-only as an option??

## May 3, 2020
Start integrating latest WSD method with word category formation.

## May 11, 2020
In the past days, integrated word categorizer with new WSD method.
Also needed to redesign some parts of the WSD senser to make it work.
Concern: Normalization of word-sense embeddings is probably needed, as
probabilities coming from different-length sentences may affect results.
