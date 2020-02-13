# Proposal to handle sub-words in sentence probability estiamtes

##Problem:
- BERT uses sub-words in its vocabulary, while the current 
sentence-probability estimator handles everything token by token, instead
of word-by word.
Currently, when a word is separated into sub-words while estimating the 
probability of the sentence, the probability increases a lot 
because each subword's probability given the neighboring sub-words
can be quite high.
E.g. 'He answered unequivocally' vs. 'He answered quickly':
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
- In addition to this, when used for word-categorization, a token
is substituted for another word. I need to avoid the substitution of
a single sub-word, and make sure a word is substituted as a whole instead.
Also, when the replacement word is composed of sub-words, I need to 
insert them all at the same time. Currently, the whole word is not tokenized
and replaced as is, which results in incorrectly calculated probability.

Proposal:
- Don't do subititution by single tokens, but always as whole words.
- For removing a token, need to keep a separate list with words, not
sub-words, and choose the one to substitute from there, then replace
all of its tokens.
- For placing a word, need to tokenize it first, and insert all tokens
together at the appropriate location
- For sentence probability estimation, if a split word is involved in
the sentence, need to mask all of its sub-words at the same time and
multiply each of the tokens' estimated value, as the estimated value of
the word. 
- For sentence probability normalization, count only words instead of 
tokens as it's currently done.