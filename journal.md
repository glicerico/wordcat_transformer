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
 - Organize code in functions
 - Try clustering with DBSCAN/OPTICS, which leave some words unclustered
 - Automate masking every word in the sentence
 - Add some voting system to decide which words go in a cluster? Currently
 all "high" words in every member of a cluster go into it.
 - Handle sub-words in sentences
 - Try with large bert model
