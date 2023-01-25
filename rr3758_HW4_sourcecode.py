import math, re, numpy as np
from collections import defaultdict

closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

def parse_query_abstract(file_location):
    
    hashmap = {}
    with open(file_location, 'r') as f:
        index = 0
        for line in f:
            if '.I' in line:
                index = index + 1
            elif '.W' not in line:
                if index not in hashmap:
                    hashmap[index]  = list(filter(None, re.split('\W|\d', line.strip()))) 
                else:
                    hashmap[index] += list(filter(None, re.split('\W|\d', line.strip())))

    f.close()

    for val in hashmap:
        temp1 = hashmap[val]
        temp2 = []

        for w in temp1:
            if w in closed_class_stop_words:
                continue
            if w.isalnum() == False:
                continue
            if w.isdigit():
                continue
            if w not in hashmap:
                temp2.append(w)

        hashmap[val] = temp2
    return hashmap

parsed_query = parse_query_abstract('cran.qry')
parsed_abstract = parse_query_abstract('cran.all.1400')

def get_query_tfidf(hash):
    hash_values = list(hash.values())
    idf = defaultdict(lambda: defaultdict(float))
    tfidf = defaultdict(lambda: defaultdict(float))
    key = 1

    for val in hash_values:
        for word in val:
            if word not in idf.keys():
                cnt = 0
                for query in hash_values:
                    if word in query:
                        cnt = cnt + 1
                idf[word] = math.log(225 / cnt)
            tf = val.count(word) / len(val)
            tfidf[key][word] = tf*idf[word]
        key = key + 1

    return tfidf

def get_abstract_tfidf(hash):
    hash_values = list(hash.values())
    idf = defaultdict(lambda: defaultdict(float))
    tfidf = defaultdict(lambda: defaultdict(float))
    key = 1

    for val in hash_values:
        for word in val:
            if word not in idf.keys():
                cnt = 0
                for query in hash_values:
                    if word in query:
                        cnt = cnt + 1
                idf[word] = math.log(1400 / cnt)
            tf = val.count(word) / len(val)
            tfidf[key][word] = tf*idf[word]
        key = key + 1

    return tfidf

query_tfidf = get_query_tfidf(parsed_query)
abstract_tfidf = get_abstract_tfidf(parsed_abstract)

query_sum = 0
abstract_sum = 0

score = defaultdict(lambda: defaultdict(float))

for key in query_tfidf.keys():

    abstract_tfidf_values = []
    query_tfidf_values = list(query_tfidf[key].values())
    query_tfidf_keys = list(query_tfidf[key].keys())

    for abstract in abstract_tfidf.keys():
        for word in query_tfidf_keys:
            if word not in list(abstract_tfidf[abstract].keys()):
                abstract_tfidf_values.append(0)
            else:
                abstract_tfidf_values.append(abstract_tfidf[abstract][word])

        for x in query_tfidf_values:
            query_sum = query_sum + x**2
        for y in abstract_tfidf_values:
            abstract_sum = abstract_sum + y**2

        dot_prod = np.dot(query_tfidf_values, abstract_tfidf_values)
        abstract_tfidf_values = []
        denominator = np.sqrt(query_sum * abstract_sum)
        similarity = 0
        
        if denominator != 0:
            similarity = dot_prod / denominator
        score[key][abstract] = similarity


for key in score:
    score[key] = {k: v for k, v in sorted(score[key].items(), key=lambda item: item[1], reverse=True)} 


with open('output.txt', 'a') as output_file:
    for query in score:
        for abstract in score[query]:
            output_file.write(str(query) + ' ' + str(abstract) + ' ' + str('{:f}'.format(score[query][abstract])) + '\n')

output_file.close()



        
        
                    
