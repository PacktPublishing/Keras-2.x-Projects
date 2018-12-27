TextData="We catched fish, and talked, and we took a swim now and then to keep off sleepiness. It was kind of solemn, drifting down the big still river, laying on our backs looking up at the stars, and we didn’t ever feel like talking loud, and it warn’t often that we laughed, only a kind of low chuckle. We had mighty good weather, as a general thing, and nothing ever happened to us at all, that night, nor the next, nor the next."

from nltk.tokenize import word_tokenize
TextDataWords=word_tokenize(TextData)
print(TextDataWords)
len(TextDataWords)

from nltk.tokenize import sent_tokenize
TextDataSent = sent_tokenize(TextData)
print(TextDataSent)
len(TextDataSent)

from nltk import pos_tag
TextDataPT = pos_tag(TextDataWords)
print(TextDataPT)
len(TextDataPT)

from nltk.probability import FreqDist
TagFrequency = FreqDist(tag for (word, tag) in TextDataPT)
print(TagFrequency.most_common())

from nltk.stem import PorterStemmer
PtSt = PorterStemmer()
for word in TextDataWords[0:20]:
    print(word + ":" + PtSt.stem(word))
    
from nltk.stem import WordNetLemmatizer
WnetLem = WordNetLemmatizer()
Plurals =['buses','wishes','pitches', 'boxes', 'pennies', 'children', 'potatoes', 'boats', 'cats', 'analyses', 'theses', 'phenomena']
for word in Plurals:
    print(word + ":" + WnetLem.lemmatize(word))