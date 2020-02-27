# wordcat-transformer
Learning word categories using pre-trained transformer embeddings.

Simple example to run word categorizer:
```
python src/word_categorizer.py --sentences sentences/1sentence.txt 
                               --vocab vocabularies/microtest.vocab 
                               --clusterer OPTICS
                               --save_to microtest 
                               --pickle_emb microtest.pickle 
                               --masks 3 --verbose
```

For full options, see code's `main` function
[documentation](src/word_categorizer.py)
