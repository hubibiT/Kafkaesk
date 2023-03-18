import os 
import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from collections import Counter
import numpy as np
import math


verarb_werke = {}
nlp = spacy.load('de_dep_news_trf')


stopwords = nlp.Defaults.stop_words
word_count = 1000

dirname = os.path.dirname(__file__)
dirnameText = os.path.join(dirname, 'corpus/')
filenames_l = [filename for filename in os.listdir(dirnameText) if not filename.startswith(".")]
filenames_list = sorted(filenames_l)
print(filenames_list)

non_unique_words_global = set(nlp.vocab.strings)


def avgPunct(doc):
    num_sentences = 0
    num_punct = 0
    for sentence in doc.sents:
        num_sentences += 1
        num_punct += len([token.text for token in sentence if ( token.pos_ == 'PUNCT' )])

    return num_punct / num_sentences


def get_max_depth(node):

    #Berechnet rekursiv die maximale Tiefe eines Parse Trees

    if node.n_lefts + node.n_rights == 0:
        # Wenn der Knoten ein Blatt ist, ist die Tiefe 1
        return 1
    else:
        # Andernfalls ist die Tiefe das Maximum der Tiefe der Kinder plus 1
        max_child_depth = 0
        for child in node.children:
            child_depth = get_max_depth(child)
            if child_depth > max_child_depth:
                max_child_depth = child_depth
        return max_child_depth + 1

def get_avg_max_depth(doc):
    max_depths = []
    for sent in doc.sents:
        max_depth = 0
        for token in sent:
            if token.dep_ == "ROOT":
                # Der Knoten des Parse Trees, der mit dem Hauptverb (ROOT) verbunden ist
                # repräsentiert den Satz. Von diesem Knoten aus kann man den maximalen
                # Verschachtelungsgrad des Parse Trees berechnen.
                depth = get_max_depth(token)
                if depth > max_depth:
                    max_depth = depth
        max_depths.append(max_depth)
    return sum(max_depths) / len(max_depths)


def Vorgangsspassiv(doc):
    count = 0
    for token in doc:
        if token.tag_ == "VVPP" and token.head.lemma_ == "werden": 
            #VVpP == Verb Partizip Passiv entspricht PartizipII
            count += 1
            #print(token)
    return count

def Zustandsspassiv(doc):
    count = 0
    for token in doc:
        if token.tag_ == "VVPP" and token.head.lemma_ == "sein":
            count += 1
            #print(token)
    return count

def konjunktivI(doc):
    count = 0
    for token in doc:
        if "Mood=Sub" in token.morph and "Tense=Pres" in token.morph:
            #Sub == Subjunctive, deutsch konjunktiv..  Bestimmung der Art über Zeitform
            count += 1
            #print(token)
    return count

def konjunktivII(doc):
    count = 0
    for token in doc:
        if "Mood=Sub" in token.morph and "Tense=Past" in token.morph:
            count += 1
            #print(token)
    return count

def rareWords(doc):
    # Anzahl seltener oder ungewöhnlicher Wörter
    non_unique_words = non_unique_words_global
    #seen_words = set()
    num_uncommon_words = 0
    for token in doc:
        if token.lemma_ not in non_unique_words: # and token.lemma_ not in seen_words:
            num_uncommon_words += 1
            #seen_words.add(token.lemma_)
            #print(token.lemma_)
    return num_uncommon_words




#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################





#txt dateien in nlp und voc dateien speichern
for filename in filenames_list:
    if filename.endswith(".txt") and not os.path.isfile(os.path.join(dirname, f'nlp_corpus/{filename}.nlp')):
        print(f"Verarbeite: {filename}")
        filepath = os.path.join(dirname, f'corpus/{filename}')
        with open(filepath) as f:
            text = f.read()
        doc = nlp(text)

        nlp_dirname = f'nlp_corpus/{filename}.nlp'
        nlp_filepath = os.path.join(dirname, nlp_dirname)
        doc.to_disk(nlp_filepath)
        vocab_dirname = f'nlp_corpus/{filename}.voc'
        vocab_filepath = os.path.join(dirname, vocab_dirname)
        doc.vocab.to_disk(vocab_filepath)



overall_most_common_words = []
counter_objs = []
words_pro_werktt = []

allSynAnalysis = []

#nlp und vocab dateien einlesen
for filename in filenames_list:
    if not filename.startswith("."):
        nlp_dirname = f'nlp_corpus/{filename}.nlp'
        nlp_filepath = os.path.join(dirname, nlp_dirname)
        verarb_werke[filename] = Doc(Vocab()).from_disk(nlp_filepath)

        vocab_dirname = f'nlp_corpus/{filename}.voc'
        vocab_filepath = os.path.join(dirname, vocab_dirname)
        verarb_werke[filename].vocab.from_disk(vocab_filepath)
        doc =  verarb_werke[filename]

        overall_most_common_words.extend([token.text for token in doc if token.pos_  not in  ['SYM' , 'PUNCT', 'SPACE']])
        counter_objs.append(Counter([token.text for token in verarb_werke[filename] if token.pos_  not in  ['SYM' , 'PUNCT' , 'SPACE']]).most_common(word_count*50))
        words_pro_werktt.append(len([token for token in verarb_werke[filename] if token.pos_  not in  ['SYM' , 'PUNCT' , 'SPACE']]))

        

        words_pro_werk = len([token for token in verarb_werke[filename] if token.pos_  not in  ['SYM' , 'PUNCT' , 'SPACE']])
        num_sentences = len(list(doc.sents))

        print(f'{filename}')
        print(words_pro_werk/num_sentences)
        print(avgPunct(doc))
        print(get_avg_max_depth(doc))
        print(Vorgangsspassiv(doc)/words_pro_werk)
        print(Zustandsspassiv(doc)/words_pro_werk)
        print(konjunktivI(doc)/words_pro_werk)
        print(konjunktivII(doc)/words_pro_werk)
        print(rareWords(doc)/words_pro_werk)
        print("\n")

        synAnalysis = []

        synAnalysis.append(words_pro_werk/num_sentences)
        synAnalysis.append(avgPunct(doc))
        synAnalysis.append(get_avg_max_depth(doc))
        synAnalysis.append(Vorgangsspassiv(doc)/words_pro_werk)
        synAnalysis.append(Zustandsspassiv(doc)/words_pro_werk)
        synAnalysis.append(konjunktivI(doc)/words_pro_werk)
        synAnalysis.append(konjunktivII(doc)/words_pro_werk)
        synAnalysis.append(rareWords(doc)/words_pro_werk)

        allSynAnalysis.append(synAnalysis)
        

allSynAnalysisNP = np.asarray(allSynAnalysis)

print(allSynAnalysisNP)
np.save(os.path.join(dirname, "data_synAnalysis21"), allSynAnalysisNP)


#print(analyze_text(verarb_werke["kafka_verwandlung.txt"]))
# indices = [733, 772, 378, 325, 908, 821, 318, 801, 184, 414, 115, 402, 118, 380, 649]
# indices1 = [864, 885, 546,  74, 923, 556, 506, 653, 109, 905, 646, 964,  11, 575, 371, 776, 673, 531,
#  615, 216, 335, 800, 242, 319, 792, 639,  46, 875, 682, 476]
# indices = [772, 947, 352, 315, 188, 376, 260, 855, 349 ,  6, 552, 149, 172, 272, 649 , 51 ,505 ,434,
#   37, 304 , 97, 761 ,477, 570 ,641 , 23 ,467 ,889, 634, 327]
# for filename in filenames_list:
#     print(filename)
#     for i in indices:
#         print(overall_most_common_words[i])


words = Counter(overall_most_common_words)
#words = Counter(overall_most_common_words[i] for i in indices)
xvals = words.most_common(word_count)
xvals = [x[0] for x in xvals]
#xvals = indices
#print(xvals)


colors= ['C0']*1 + ['C1'] * 3 + ['C2']*4 + ['C4']*4 + ['C5']*3 + ['C6']*2 + ['C7']*3 +['C8']*1
#words_pro_werk = []


all_yVals = []
#yvals = []
#i = -1
#print(zip(colors, os.listdir(dirnameText)))
# #for filename in os.listdir(dirnameText):
for i, (color, filename) in enumerate(zip(colors, filenames_list)):
     if not filename.startswith("."):
        
        #i=i+1
        yvals = []
        #words_pro_werk.append(len(verarb_werke[filename]))

        keys = [x[0] for x in counter_objs[i]]

        values = [x[1] for x in counter_objs[i]]

        for val in xvals:

            if val in keys:
                yvals.append(values[keys.index(val)]/float(words_pro_werktt[i]))

            else:
                yvals.append(0)

        # print(filename)
        # for  ind in indices:
        #     print(overall_most_common_words[ind])
        #     print(yvals[ind])


        all_yVals.append(yvals)

        print( i, (color, filename))


All_yvals = np.asarray(all_yVals)
#print(All_yvals)
np.save(os.path.join(dirname, "data_wordlength21"), All_yvals)
# plt.title("Test")
# plt.xticks(rotation=90)
# plt.show()
# plt.close()



# #STOPPPPPPP


# #plt.savefig(os.path.join(dirname, f'Plotsdh1/test.png'))


# #print(spacy.explain("VM"))

# #print(avgSentLength(verarb_werke['verwandlung']))



# #To read the data again:
# #verarb_werke['verwandlung'] = Doc(Vocab()).from_disk(nlp_filepath)
# #verarb_werke['verwandlung'].vocab.from_disk(vocab_filepath)
# #for token in text:
#     #print(token.text, token.pos_, token.dep_)


# #print("Noun phrases:", [chunk.text for chunk in text.noun_chunks])
# #print("Nouns:", [token.text for token in text if (token.pos_ == "NOUN") or (token.pos_ == "PROPN")])

# #verben = [token.lemma_ for token in verarb_werke['verwandlung'] if (token.pos_ == "VERB")]
# #word_freq = Counter(verben)
# #print("Häufigste Nomen von:")
# #for word, freq in word_freq.most_common(10):
#     #print("{}: {}".format(word, freq))


# # Find named entities, phrases and concepts
# #for entity in text.ents:
#     #print(entity.text, entity.label)

# ###fancier/easier plot in matplotlib

# #xvals = [word for word, _ in word_freq.most_common(10)]
# #yvals = [count for _,count in word_freq.most_common(10)]

# #fig=plt.figure(figsize=(4,3))
# #plt.bar(xvals, yvals, color='k', width=0.4)
# #plt.xlabel("X axis label")
# #plt.ylabel("Y axis label")
# #plt.show()