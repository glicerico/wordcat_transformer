Here's a sample of the word category formation results using words as 
vectors, where each vector dimension is the probability of a given 
sentence (with a blank in a fixed position) when the current word is 
placed in the blank location, as Ben suggested.
The vocabulary to categorize is [here](../vocabularies/POS_unambiguous.vocab),
and the sentences used to create the features are 
[here](../sentences/19sentences.txt).
The labels in the vocabulary were used to evaluate the clustering
(using F1 measure).

The vectors seem expressive, and even with a relatively small number 
of dimensions (163, corresponding to only 19 simple sentences with 
blanks in different locations) the categories are not bad.

I have tried a few clustering methods. The best results for clustering 
all words are given by regular K-means. Most clusters are good, 
with the occasional one or two clusters being pretty bad 
(see cluster 8 in this example):
```
Cluster #0:
[airline, audience, australia, baby, banana, band, bed, bird, book, building, cat, clock, country, dog, family, flowers, fly, happiness, history, house, love, man, match, milk, mountain, movie, music, ocean, phone, photograph, rain, rice, sleep, snow, socks, state, sunlight, train, village, violin, wealth, work, ]
Cluster #1:
[and, as, because, but, if, or, until, when, where, which, while, yet, ]
Cluster #2:
[anyone, badly, brutally, carefully, eagerly, extremely, few, grimly, happily, never, none, quietly, quite, seem, themselves, urgently, whoever, whom, who, ]
Cluster #3:
[am, are, become, been, be, can, did, has, have, is, may, must, should, than, was, will, ]
Cluster #4:
[her, his, my, some, these, this, ]
Cluster #5:
[cut, do, drive, eat, go, grow, make, play, stop, swim, walk, write, ]
Cluster #6:
[above, across, around, at, behind, below, beside, between, by, during, from, off, of, on, onto, through, to, under, with, ]
Cluster #7:
[christian, him, i, it, me, myself, santiago, she, them, they, we, you, yourself, yours, ]
Cluster #8:
[away, daily, everywhere, eyeglasses, here, kilimanjaro, lazily, most, once, quickly, something, somewhere, soon, there, though, too, up, upstairs, very, weekly, well, what, yesterday, ]
Cluster #9:
[asian, bad, clean, cold, dry, entire, final, french, funny, good, hot, little, organized, pregnant, pretty, sharp, ]
```
This is probably coming from the K-means condition that every word must
 be placed in a cluster. 
Actually, words like `eyeglasses`, `kilimanjaro` and `lazily` are 
multi-token (decomposed into sub-words) and I currently don't handle 
them properly (see below).

In an attempt to avoid this stray-word problem, I tried a couple 
out-of-the-box methods that do not necessarily cluster all words:
Sklearn's [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)
doesn't cluster words in less-dense regions of the space, and it
is very sensitive to its density parameters. 
It only performs non-trivial clustering
in a very narrow parameter margin which I found using the 
F1-metric against the gold standard; so it doesn't seem very useful
for our unsupervised problem.
DBSCAN seems suited for clusters with homogeneous density.

In contrast, [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS)
also from the sklearn library, works okay with default settings.
It uses a reachability criterion which allows for clusters of different
density.
The clusters obtained are of high quality, and perhaps a bit too 
fine-grained sometimes (compare below clusters 9, 11, 14, 15, 17, and 18, each 
gathering different types of pronouns).
The problem is that most of the words are
actually unclustered (cluster -1).
I wonder if more features (more sentences) would help the clustering 
algorithm to group them.
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

These are some changes I'm hoping will improve the results:
- I need to handle polysemy beforehand, probably using the BERT-based 
approach I attempted earlier. 
This way, I can build vectors for each word-sense and perhaps avoid 
some of the mis-categorization coming from this.
- BERT uses sub-words in its vocabulary, and currently my code handles everything token by token. So I need to handle word by word instead. Currently, when a word is separated into sub-words, the probability of the sentence increases a lot because each subword's probability is quite high, given the neighboring sub-words. E.g. 'He answered unequivocally' vs. 'He answered quickly'
```
Processing sentence: ['[CLS]', 'he', 'answered', 'une', '##qui', '##vo', '##cal', '##ly', '.', '[SEP]']
Word: he         Prob: 0.2814375162124634
Word: answered   Prob: 0.006721243727952242
Word: une        Prob: 0.9973625540733337
Word: ##qui      Prob: 0.9999865293502808
Word: ##vo       Prob: 0.9999856948852539
Word: ##cal      Prob: 0.9999865293502808
Word: ##ly       Prob: 0.9979932308197021
Word: .          Prob: 0.9998167157173157

Normalized sentence prob: log(P(sentence)) / sent_length: -0.784400124636818

Processing sentence: ['[CLS]', 'he', 'answered', 'quickly', '.', '[SEP]']
Word: he         Prob: 0.2151750773191452
Word: answered   Prob: 0.026344342157244682
Word: quickly    Prob: 0.05330450460314751
Word: .          Prob: 0.9981406927108765

Normalized sentence prob: log(P(sentence)) / sent_length: -2.0266001676791348
```
- Still wondering if I should implement the Clark-like clustering 
previously used for WSD. Re-reading that previous code, I remembered 
that I need a parameter to decide when two clusters are close enough
to merge them. This may be equivalent to the DBSCAN density parameter,
and also not be suitable for a varying density space like the one I
think we have here. I think the 2 issues above are more pressing now.

