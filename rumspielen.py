import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.pipeline import morphologizer
from spacy import displacy
import numpy as np


nlp = spacy.load('de_dep_news_trf')
# morphologizer = morphologizer()
# nlp.add_pipe(morphologizer)


#text = "Peter kam zu spät. Er sagte, dass der Zug Verspätung hätte. Er sagte, der Zug habe Verspätung gehabt."
#text = "Unser Lehrer sagt, wir müssten noch viel lernen. Ich wünschte, ich hätte Ferien. Wenn ich im Urlaub wäre, läge ich den ganzen Tag am Strand. Wärst du so freundlich, an die Tafel zu kommen? Wenn ich reich wäre, würde ich mir ein Schloss   kaufen. Er habe zu viel getrunken, meinte der Opa."
#text = "Wenn der Zug pünktlich gewesen wäre, wäre Peter wohl rechtzeitig gekommen. Hoffentlich fährt der Zug pünktlich, denn dann käme Peter rechtzeitig. Peter würde vielleicht kommen, wenn der Zug heute fährt. Peter kam hereingerannt als ob er den Zug verpasst habe."
text = "Jemand musste Josef K. verleumdet haben, denn ohne daß er etwas Böses getan hätte, wurde er eines Morgens verhaftet. Die Köchin der Frau Grubach, seiner Zimmervermieterin, die ihm jeden Tag gegen acht Uhr früh das Frühstück brachte, kam diesmal nicht. Das war noch niemals geschehen. Er sagt, sie gingen joggen. Er sagte, sie seien im Kino."
# text = "Ich heiße Leopold und esse gerne Erdbeereis, weil es lecker ist."
#text = "Die Tür wurde von ihm geöffnet und das Haus wird renoviert." #Ein Mann wurde angefahren. Er ist verletzt." #Dem Verletzten wurde ein Verband angelegt. Jetzt wird der Mann ins Krankenhaus gebracht."
#text = "Der Mann ist verletzt. Der Mann ist verletzt gewesen. Der Mann war verletzt. Der Mann war verletzt gewesen. Der Mann wird verletzt sein. Der Mann wird verletzt gewesen sein. Der Mann wird verletzt. Der Mann ist verletzt worden. Der Mann wurde verletzt. Der Mann war verletzt worden. Der Mann wird verletzt werden. Der Mann wird verletzt worden sein."
#text = 'Der Held kämpft tapfer.'
doc = nlp(text)
for token in doc:
    print("Token Text: ", token.text)
    print("Token POS: ", token.pos_)
    print("Token Tag:", token.tag_)
    print("Token dep_: ", token.dep_)
    print("Part of Speech: ", token.pos_)
    print("Morphologizer: ", token.morph)
    print("Lemma: ", token.lemma_)
    print("Head Token: ", token.head.text)
    print("\n")

count = 0
for token in doc:
    if "Mood=Sub" in token.morph and "Tense=Past" in token.morph:
        count += 1
        print(token)
print(f"Anzahl der Konjunktiv-II-verben: {count}")

count = 0
for token in doc:
    if "Mood=Sub" in token.morph and "Tense=Pres" in token.morph:
        count += 1
        print(token)
print(f"Anzahl der Konjunktiv-I-verben: {count}")



def get_max_depth(node):
    """
    Berechnet rekursiv die maximale Tiefe eines Parse Trees
    """
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
    """
    Berechnet den durchschnittlichen maximalen Verschachtelungsgrad aller Sätze im Text
    """
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


# Berechne den durchschnittlichen maximalen Verschachtelungsgrad
avg_max_depth = get_avg_max_depth(doc)

print(f"Durchschnittlicher maximaler Verschachtelungsgrad: {avg_max_depth}")


count = 0
for token in doc:
    if token.tag_ == "VVPP" and token.head.lemma_ == "werden":
        count += 1
        #print(token)
print(f"Vorgangspassiv: {count}")

count = 0
for token in doc:
    if token.tag_ == "VVPP" and token.head.lemma_ == "sein":
        count += 1
        #print(token)
print(f"Zustandspassiv: {count}")


svg = displacy.render(doc, style="dep")
with open('/Users/hubitragenap/Documents/Uni/Python/DIHU1/deptree.png', "w", encoding="utf-8") as file:
            file.write(svg)
np.save('/Users/hubitragenap/Documents/Uni/Python/DIHU1/deptree.png', svg)


print(doc.print_tree(light=True))






