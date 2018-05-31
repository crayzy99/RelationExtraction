import os
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordParser
from nltk.stem.snowball import SnowballStemmer
from nltk.collocations import *
from collections import Counter

def load_ieee_keywords(path='./data/ieee'):
    keywords = set()
    for line in open(path, encoding='utf-8'):
        keywords.add(line.strip().lower())
    print('Loaded %d ieee taxonomy keywords' % len(keywords))
    return frozenset(keywords)

punc = frozenset(string.punctuation + '–')
stoplist = frozenset(stopwords.words('english'))
stemmer = SnowballStemmer('english')
st, chunker, chunker_compl = None, None, None
parser = None
# simple trick, store two maps in one dict
label_int_bimap = {'None': 0, 'Material': 1, 'Process': 2, 'Task': 3, \
                0: 'None', 1: 'Material', 2: 'Process', 3: 'Task'}
ieee_keywords = load_ieee_keywords()
wiki_freq, wiki_idf = None, None
glove_obj = None
glove_cache = {}

def label2int(label):
    assert(isinstance(label, str) and label in label_int_bimap)
    return label_int_bimap[label]

def int2label(idx):
    idx = int(idx)
    assert(idx >= 0 and idx <= 3 and isinstance(idx, int))
    return label_int_bimap[idx]

def is_valid_label(label):
    return label2int(label) > 0

def is_punc(ch):
    return (ch in punc)

def all_punc(w):
    return all(is_punc(ch) for ch in w)

def no_char(word):
    return not any((c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z') for c in word)

def is_stopword(word):
    return (word.lower() in stoplist)

def remove_punctuation(word_list):
    return ''.join(map(lambda c: not is_punc(c), word_list))

def has_digit(s):
    return any(c.isdigit() for c in s)

def has_punc(s):
    return any(is_punc(c) for c in s)

def all_zero(arr):
    return all(abs(e) < 1e-5 for e in arr)

def has_common_word(w1, w2, remove_punc=True, remove_stop=True, remove_digit=True):
    t1 = get_token_list(w1, remove_punc=remove_punc, remove_stop=remove_stop, remove_digit=remove_digit)
    t2 = get_token_list(w2, remove_punc=remove_punc, remove_stop=remove_stop, remove_digit=remove_digit)
    t1 = map(lambda e: e.lower(), t1)
    t2 = map(lambda e: e.lower(), t2)
    return len(set(t1) & set(t2)) > 0

def get_token_list(text, remove_punc=False, remove_stop=False, remove_digit=False):
    assert(isinstance(text, str))
    token_list = nltk.word_tokenize(text)
    if remove_punc:
        token_list = filter(lambda c: not is_punc(c), token_list)
    if remove_stop:
        token_list = filter(lambda c: not is_stopword(c), token_list)
    if remove_digit:
        token_list = filter(lambda c: not c.isdigit(), token_list)
    return list(token_list)

def get_namedentities(text):
    global st
    if not st:
        st = StanfordNERTagger('utils/english.conll.4class.caseless.distsim.crf.ser.gz','utils/stanford-ner.jar')
    tokens = get_token_list(text, remove_punc=True, remove_digit=True)
    ner_tagged = st.tag(tokens)
    assert(len(tokens) == len(ner_tagged))
    named_entities = []
    l, r = 0, 0
    while l < len(ner_tagged):
        if ner_tagged[l][1] == 'O':
            l += 1
        else:
            r = l + 1
            while r < len(ner_tagged) and ner_tagged[r][1] == ner_tagged[l][1]:
                r += 1
            named_entities.append(' '.join(tokens[l:r]))
            l = r
    return named_entities

def del_citation(text):
    while True:
        p1 = text.find('[')
        p2 = text.find(']', p1)
        if p1 < 0 or p2 < 0:
            return text
        else:
            text = text[:p1] + text[(p2+1):]

def get_nounphrases(text):
    global chunker, chunker_compl
    if not chunker:
        # better candidate coverage, but lower validation performance
        # grammar = r"""
        #     NBAR:
        #         {<NN|NNS|NNP|NNPS|JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>}

        #     NP:
        #         {<NBAR><IN><NBAR>}
        # """
        grammar = r"""
            NBAR:
                {<NN|NNS|NNP|NNPS|JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>}

            NP:
                {<NBAR><IN><NBAR>}
                {<NBAR>}
        """
        # this pattern gives similar performance
        # grammar = r"""
        # NP:
        #     {<NN|NNS|NNP|NNPS|JJ>*<NN|NNS|NNP|NNPS|VBG>}
        # """
        chunker = nltk.RegexpParser(grammar)
        grammar_compl = r"""
            VBAR:
                {<NN><IN><DT><NN>}
                {<VBG><NN><NN>}
                {<JJ><CC><JJ><NNS>}
                {<VBG><DT><JJ><NN>}
                {<VBG><NN><NNS>}
                {<VB><DT><JJ><NN>}
                {<NN><VBG><NN>}
                {<VB><DT><NN><NN>}
                {<JJ><JJ><NNS>}
                {<NN><JJ><NN>}
                {<RB><JJ><NNS>}
                {<VBG><JJ><NNS>}
                {<DT><JJ><NN><NN>}
                {<RB><VBN><NN>}
                {<VBN><JJ><NN>}
                {<RB><VBN><NNS>}
                {<NN><VBN><NN>}
                {<VBN><JJ><NNS>}
                {<NNS><IN><DT><NN>}
        """
        chunker_compl = nltk.RegexpParser(grammar_compl)
    text = del_citation(text)
    tokens = get_token_list(text)
    sent = nltk.pos_tag(tokens)

    noun_phrases = set()
    cand_label_set = frozenset(['NBAR'])
    for chk in [chunker]:
        tree = chk.parse(sent)
        # print(tree)
        for subtree in tree.subtrees():
            if subtree.label() in cand_label_set:
                phrase = ' '.join(map(lambda t: t[0], subtree.leaves())) # <word, tag> tuple
                noun_phrases.add(phrase)

    return list(noun_phrases)

def get_stanford_nounphrases(sentences):
    global parser
    if not parser:
        print('Instantiate stanford parser...')
        parser = StanfordParser('./utils/stanford-parser.jar', './utils/stanford-parser-3.6.0-models.jar')
    sents = list(map(lambda s: s.sent, sentences))
    trees = list(parser.raw_parse_sents(sents))
    noun_phrases = set()
    for tree in trees:
        tree = list(tree)[0]
        # print(tree)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                phrase = ' '.join(subtree.leaves())
                noun_phrases.add(phrase)
    return list(noun_phrases)

def __get_ngrams_aux(tokens, n=2):
    ret = Counter()
    for i in range(len(tokens) - n + 1):
        word = ' '.join(tokens[i:i + n])
        ret[word] += 1
    return ret

def get_ngrams(text, n=2):
    tokens = get_token_list(text)
    cnt = Counter()
    l, r = 0, 0
    while r < len(tokens):
        if has_digit(tokens[r]) or has_punc(tokens[r]) or is_stopword(tokens[r]):
            cnt += __get_ngrams_aux(tokens[l:r], n=n)
            r += 1
            l = r
        else:
            r += 1
    cnt += __get_ngrams_aux(tokens[l:r], n=n)
    return list(cnt.keys())

def is_ieee_keyword(word):
    return (word.lower() in ieee_keywords)

def get_ieee_keywords():
    for kw in ieee_keywords:
        yield kw

def get_occur_positions(word, text, return_on_hit=False):
    assert(isinstance(word, str) and isinstance(text, str))
    if not word or not text:
        return []
    occur_pos = []
    ptr = 0
    while ptr < len(text):
        p = text.find(word, ptr)
        if p < 0:
            break
        if (p - 1 >= 0 and text[p - 1].isalpha()) \
                or (p + len(word) < len(text) and text[p + len(word)].isalpha()):
            ptr = p + len(word)
            continue
        occur_pos.append(p)
        if return_on_hit:
            return occur_pos
        ptr = p + len(word)
    return occur_pos

def has_occur(word, text):
    return len(get_occur_positions(word, text)) > 0

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def get_tag_list(tokens):
    if not isinstance(tokens, list):
        tokens = word_tokenize(tokens)
    word_pos_tp = pos_tag(tokens)
    return list(map(lambda e: e[1], word_pos_tp))

def get_model(model_name):
    import xgboost as xgb

    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    if model_name == 'svm':
        return svm.LinearSVC()
    elif model_name == 'svc':
        return svm.SVC(probability=True, kernel='linear')
    elif model_name == 'gb':
        return GradientBoostingClassifier(n_estimators=100)
    elif model_name == 'gbr':
        return GradientBoostingRegressor(n_estimators=100)
    elif model_name == 'lr':
        return LogisticRegression()
    elif model_name == 'xgb':
        max_depth = get_rnd_element([4])
        learning_rate = get_rnd_element([0.01])
        n_estimators = get_rnd_element([200])
        gamma = get_rnd_element([0])
        min_child_weight = get_rnd_element([1.0])
        reg_alpha = get_rnd_element([0.3])
        reg_lambda = get_rnd_element([0])
        scale_pos_weight = get_rnd_element([1])
        print('max_depth: %d, learning_rate: %f, n_estimators: %d, \
            gamma: %f, min_child_weight: %f, reg_alpha: %f, reg_lambda: %f, scale_pos_weight: %f\n' % \
            (max_depth, learning_rate, n_estimators, gamma, min_child_weight, reg_alpha, reg_lambda, scale_pos_weight))
        return xgb.XGBClassifier(max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=min_child_weight)
    elif model_name == 'rfr':
        return RandomForestRegressor(n_estimators=200, n_jobs=-1)
    else:
        # return ExtraTreesClassifier(n_estimators=500, n_jobs=-1, oob_score=True, bootstrap=True)
        return RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True)

def get_rnd_element(l):
    from random import randint
    assert(isinstance(l, list))
    return l[randint(0, len(l) - 1)]

def cosine(a, b):
    num = sum(a * b)
    if abs(num) < 1e-4:
        return 0.0
    return num / np.sqrt(sum(a * a)) / np.sqrt(sum(b * b))

def subfinder(mylist, pattern):
    assert(len(pattern) > 0)
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:(i + len(pattern))] == pattern:
            matches.append(pattern)
    return matches

def __load_counter(path):
    ret = Counter()
    for line in open(path, encoding='utf-8'):
        fs = line.strip().split('\t')
        if len(fs) == 0:
            continue
        k, v = fs[0], fs[1]
        ret[k] = float(v)
    return ret

def get_wiki_freq(word, relative=True):
    global wiki_freq
    if not wiki_freq:
        wiki_freq = __load_counter(path='./data/freq')
    word = word.lower()
    if word not in wiki_freq:
        return 0
    ret = wiki_freq[word]
    return ret / wiki_freq['the'] if relative else ret

def get_wiki_idf(word):
    global wiki_idf
    if not wiki_idf:
        wiki_idf = __load_counter(path='./data/idf')
    word = word.lower()
    if word not in wiki_idf:
        return 20
    else:
        return wiki_idf[word]

def get_wiki_idf_dict():
    global wiki_idf
    if not wiki_idf:
        wiki_idf = __load_counter(path='./data/idf')
    return dict(wiki_idf)

def get_word_tf(word, doc):
    return len(get_occur_positions(word.lower(), doc.text.lower()))

def dump_feature_importance(names, importances, path):
    assert(len(names) == len(importances))
    writer = open(path, 'w', encoding='utf-8')
    cnt = Counter({k: v * 10**6 for k, v in zip(names, importances)})
    for k, v in cnt.most_common():
        writer.write('%s\t%f\n' % (k, v))
    writer.close()
    return cnt

def get_embedding(words):
    from glove import Glove
    global glove_obj
    if not glove_obj:
        glove_obj = Glove()
    if words not in glove_cache:
        glove_cache[words] = glove_obj.get_vector(words)
    return glove_cache[words]

def get_stem_suffix(word):
    stem_word = stemmer.stem(word)
    ptr = 0
    while ptr < min(len(stem_word), len(word)) and word[ptr] == stem_word[ptr]:
        ptr += 1
    suffix = word[ptr:]
    if len(suffix) == 0:
        suffix = 'NOSUFFIX'
    return stem_word, suffix

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

# SVD for dimensionality reduction
class SVDUtil:

    def __init__(self, corpus, df=1, path='./data/svd'):
        if os.path.exists(path):
            print('Load svd from cache...')
            self.vectorizer, self.svd = joblib.load(path)
        else:
            print('Fit svd transformer...')
            self.vectorizer = CountVectorizer(min_df=df)
            self.svd = TruncatedSVD(n_components=20, random_state=42)
            self._fit(corpus)
            joblib.dump((self.vectorizer, self.svd), path)

    def _fit(self, corpus):
        paras = []
        for doc in corpus:
            paras.extend([self._tokenize(para) for para in doc.all_text])
            paras.append(self._tokenize(doc.abstract))
        X = self.vectorizer.fit_transform(paras)
        self.svd.fit(X)

    def get(self, text):
        transformed_text = self.vectorizer.transform([self._tokenize(text)])
        return self.svd.transform(transformed_text)[0]

    def _tokenize(self, text, remove_citation=True):
        if remove_citation:
            text = del_citation(text)
        tokens = get_token_list(text, remove_stop=True, remove_punc=True, remove_digit=True)
        return ' '.join(tokens).lower()


def get_capital_words(sent):
    ret = set()
    tokens = get_token_list(sent)
    # add words with only capital letters or digits
    for token in tokens:
        if token.upper() == token and token[0] >= 'A' and token[0] <= 'Z':
            ret.add(token)
    # add words which start with capital letters
    ptr = 0
    while ptr < len(tokens):
        i = ptr
        while i < len(tokens) and len(tokens[i]) > 0 and tokens[i][0].isupper():
            i += 1
        if i > ptr and not (ptr == 0 and i == 1):
            word = ' '.join(tokens[ptr:i])
            ret.add(word)
            if i + 1 < len(tokens) and get_tag_list(tokens[i+1:i+2])[0].startswith('NN'):
                ret.add(word + ' ' + tokens[i + 1])
        ptr = i + 1
    return list(ret)

def get_parenthesis_words(sent):
    ret = set()
    start = 0
    while True:
        l = sent.find('(', start)
        r = sent.find(')', l + 1)
        if l < 0 or r < 0:
            break
        word = sent[l+1:r]
        start = r + 1
        if word.upper() == word and all(c >= 'A' and c <= 'Z' for c in word):
            tokens = get_token_list(sent[:l])
            ret.add(' '.join(tokens[-len(word):]))
    return list(ret)
