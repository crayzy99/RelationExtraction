
import nltk.data
import os
import json

from util import xml2Doc
from nlp_util import *
from collections import Counter

class Document:

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def __init__(self, path, data_type='train', text=None):
        '''
            Read from raw text and separate it to sentences.
            '''
        if not path.endswith('.txt'):
            assert False
        self.path = path
        self.xml_path = path.replace('.txt', '.xml')
        self.ann_path = path.replace('.txt', '.ann')
        self.cand_words = None #
        self.score_cache = {} #
        self.title = None
        self.highlights = None
        self.abstract = None
        self.all_text = None
        self.keyphrases = [] # NEW
        self.kp_position = {} # NEW
        self.kp_cnt = {}
        if not text:
            self.__load_from_cache_or_parse() # ???
            reader = open(self.path, encoding='utf-8')
            for line in reader:
                self.text = line.strip()  # only one line
            reader.close()
        else:
            self.text = text
        assert(len(self.text) > 0)

        self.sentences = []
        self.sent_offset = [] #offsets of each sentence in self.sentences
        ptr = 0
        for sent in Document.sent_detector.tokenize(self.text): # list of sentences
            start = self.text.find(sent, ptr) #find the sentence in raw text
            assert(start >= 0 and start >= ptr)
            if len(self.sentences) > 0 and \
                (self.sentences[-1].sent.endswith('e.g.') or self.sentences[-1].sent.endswith('i.e.')
                    or self.sentences[-1].sent.endswith('al.') or self.sentences[-1].sent.endswith('Fig.')
                    or sent[0].islower()):
                self.sentences[-1] = Sentence(self.text[self.sent_offset[-1]: (start + len(sent))], self)
            else:
                self.sentences.append(Sentence(sent, self))
                self.sent_offset.append(start)
            ptr = start + len(sent)

    def add_global_keyword(self, word, label='Material'):
        for sent_obj in self.sentences:
            sent_obj.add_keyword(word=word, label=label)

    def add_global_relation(self, arg1, arg2, label):
        assert(label != None and label != 'None')
        for sent_obj in self.sentences:
            sent_obj.add_relation(arg1=arg1, arg2=arg2, label=label)

    def write2ann(self, path):
        writer = open(path, 'w', encoding='utf-8')
        tag_cnt, rel_cnt = 1, 1
        word2tag = {}
        for offset, sent_obj in zip(self.sent_offset, self.sentences):
            sorted_keywords = sent_obj.get_sorted_keywords(duplicate=True)
            printed = set()
            for i, (keyword, pos) in enumerate(sorted_keywords):
                if i > 0:
                    pv_word, pv_pos = sorted_keywords[i - 1]
                    assert(pos > pv_pos or (pos == pv_pos and len(pv_word) >= len(keyword)))
                    if pos <= pv_pos + len(pv_word):
                        continue
                if keyword in printed:
                    continue
                word2tag[keyword] = 'T' + str(tag_cnt)
                start = offset + pos
                end = start + len(keyword)
                writer.write('%s\t%s %d %d\t%s\n' % \
                                (word2tag[keyword], self.get_label(keyword, form='str'), start, end, self.text[start:end]))
                tag_cnt += 1
                printed.add(keyword)
            for rel in sent_obj.relations:
                if not (rel.arg1 in word2tag and rel.arg2 in word2tag):
                    continue
                if rel.label == 'Synonym-of':
                    writer.write('*\tSynonym-of %s %s\t\n' % (word2tag[rel.arg1], word2tag[rel.arg2]))
                elif rel.label == 'Hyponym-of':
                    writer.write('R%d\tHyponym-of Arg1:%s Arg2:%s\t\n' % (rel_cnt, word2tag[rel.arg1], word2tag[rel.arg2]))
                else:
                    assert False
                rel_cnt += 1

        writer.close()

    def load_ann_files(self, data_type='train'):
        tag2word = {}
        for line in open(self.ann_path, encoding='utf-8'):
            # parse field
            # An example of lines in ann_files:
            #   T3    Material 127 144    amorphous alumina
            fs = line.strip().split('\t')
            assert(len(fs) == 2 or len(fs) == 3)
            tag, label_position = fs[0], fs[1]
            if len(fs) == 3:
                word = fs[2]

            if tag.startswith('T'): # a keyphrase
                tag2word[tag] = word
                if label_position.find(';') >= 0: # data is inconsistent, hack code
                    label_position = label_position.split(';')[0]
                label, start, end = label_position.split()
                start, end = int(start), int(end)
                sent_pos = self.__binary_search_sent(start, end)
                sent_obj = self.sentences[sent_pos] # locate the sentence with this key phrase
                sent_obj.add_keyword(word=word, label=label)
                #####Begin#####
                self.keyphrases.append(word)
                self.kp_position[word] = [] 
                self.kp_cnt[word] = 0
                for i in range(len(self.sentences)):
                    sent_obj = self.sentences[i]
                    if sent_obj.sent.find(word) > 0:
                        self.kp_position[word].append(i)
                        self.kp_cnt[word] += 1
                #####End#####
            elif (tag.startswith('R') or tag == '*') and data_type == 'train': # a relation
                if len(label_position.split()) > 3:
                    print('Found more than 3 field in %s line: %s' % (self.ann_path, line.strip()))
                label, arg1, arg2 = label_position.split()[:3]
                assert(label == 'Hyponym-of' or label == 'Synonym-of')
                arg1 = tag2word[arg1.split(':')[-1]]
                arg2 = tag2word[arg2.split(':')[-1]]
                for sent_obj in self.sentences:
                    sent = sent_obj.sent
                    if sent.find(arg1) >= 0 and sent.find(arg2) >= 0:
                        sent_obj.add_relation(arg1, arg2, label)
            else:
                pass

    def is_keyword(self, word, strict_mode=True):
        for sent_obj in self.sentences:
            if sent_obj.is_keyword(word=word, strict_mode=strict_mode):
                return True
        return False

    def clear_keywords(self):
        for sent_obj in self.sentences:
            sent_obj.clear_keyword()

    def get_label(self, word, form='int'):
        label_str = 'None'
        for sent_obj in self.sentences:
            if sent_obj.is_keyword(word=word):
                label_str = sent_obj.get_label(word)
                break
        return label2int(label_str) if form == 'int' else label_str

    def get_ranking_score(self, word):
        if word not in self.score_cache:
            if self.is_keyword(word):
                self.score_cache[word] = 1
            else:
                v_word = get_embedding(word)
                if all_zero(v_word):
                    score = 0
                else:
                    keyword_embeddings = [get_embedding(keyword) for keyword in self.__get_all_keywords()]
                    keyword_embeddings = filter(lambda e: not all_zero(e), keyword_embeddings)
                    match_score = [cosine(v_word, embedding) for embedding in keyword_embeddings]
                    if len(match_score) < 2:
                        score = 0
                    else:
                        score = sorted(match_score)[-2] / 2
                self.score_cache[word] = score
        return self.score_cache[word]

    def get_keywords_with_label(self, label='Task'):
        # only for debug purpose
        for sent_obj in self.sentences:
            for keyword in sent_obj.keywords:
                if keyword.label == label:
                    yield keyword.word

    def __get_all_keywords(self):
        ret = set()
        for sent_obj in self.sentences:
            for keyword in sent_obj.keywords:
                ret.add(keyword.word)
        return ret

    def __load_from_cache_or_parse(self):
        json_path = self.path.replace('.txt', '.json')
        # avoid parsing XML every time we run code
        if not os.path.exists(json_path):
            self.title, self.highlights, self.abstract, self.all_text = xml2Doc(self.xml_path)
            s = json.dumps({'title': self.title, 'highlights': self.highlights, \
                'abstract': self.abstract, 'all_text': self.all_text}, \
                ensure_ascii=False, sort_keys=True, indent=4)
            writer = open(json_path, 'w', encoding='utf-8')
            writer.write(s)
            writer.close()
        else:
            reader = open(json_path, encoding='utf-8')
            obj = json.load(reader)
            reader.close()
            self.title, self.highlights, self.abstract, self.all_text = \
                obj['title'], obj['highlights'], obj['abstract'], obj['all_text']

    def __binary_search_sent(self, start, end):
        # binary search to find out which sentence contains [start, end)
        assert(start < end and end <= len(self.text))
        l, r = 0, len(self.sentences) - 1
        while l < r:
            mid = (l + r) // 2
            sent_l, sent_r = self.sent_offset[mid], self.sent_offset[mid] + len(self.sentences[mid].sent)
            if sent_l <= start and sent_r >= end:
                return mid
            elif sent_l > start:
                r = mid - 1
            elif sent_r <= start:
                l = mid + 1
            else:
                print('[%d, %d] vs [%d, %d] %s %s' % (start, end, sent_l, sent_r, self.text[start:end], self.text[sent_l:sent_r]))
                return mid
        return l

class Sentence:

    def __init__(self, sent, doc):
        self.sent = sent
        assert(len(self.sent) > 0)
        self.tokens = get_token_list(self.sent)
        self.doc = doc
        self.keywords = set()  # a set of Keyword objects
        self.relations = set() # a set of Relation objects
        self.ws = set()
        self.__build_tokens_offset()

    def clear_keyword(self):
        self.keywords = set()
        self.ws = set()

    def is_keyword(self, word, strict_mode=True):
        if word in self.ws:
            return True
        if not strict_mode:
            for keyword in self.keywords:
                if has_occur(word, keyword):
                    return True
        return False

    def get_seq_label(self):
        ret = [0 for _ in self.tokens]
        for keyword in self.keywords:
            word, label_int = keyword.word, label2int(keyword.label)
            pos = self.sent.find(word)
            if pos >= 0:
                start, end = pos, pos + len(word)
                for i, token in enumerate(self.tokens):
                    offset = self.token_offset[i]
                    if offset >= start and offset <= end:
                        ret[i] = label_int
        return ret

    def get_label(self, word):
        assert(self.is_keyword(word, strict_mode=True))
        for keyword in self.keywords: # not quite efficient, but that's ok.
            if keyword.word == word:
                return keyword.label
        assert False

    def add_keyword(self, word, label='Material'):
        if has_occur(word, self.sent):
            self.ws.add(word)
            self.keywords.add(Keyword(sent_obj=self, word=word, label=label))

    def add_keyword_by_pos(self, start, end, label='Material'):
        assert(start < end and start >= 0 and end <= len(self.token_offset))
        l, r = self.token_offset[start], self.token_offset[end - 1] + len(self.tokens[end - 1])
        self.add_keyword(word=self.sent[l:r], label=label)

    def add_relation(self, arg1, arg2, label):
        assert(label != None and label != 'None')
        if arg1 in self.ws and arg2 in self.ws:
            self.relations.add(Relation(sent_obj=self, arg1=arg1, arg2=arg2, label=label))

    def get_sorted_keywords(self, duplicate=False):
        # get list of <keyword, position>,
        # fist sort by position in ascending order,
        # then sort by length of word in descending order.
        word_pos = []
        for keyword in self.keywords:
            word = keyword.word
            occur_pos = get_occur_positions(word, self.sent, return_on_hit=(not duplicate))
            assert(len(occur_pos) > 0)
            if not duplicate:
                occur_pos = occur_pos[:1]
            for pos in occur_pos:
                word_pos.append((word, pos))
        word_pos = sorted(word_pos, key=lambda e: (e[1], -len(e[0])))
        return word_pos

    def has_relation(self, w1, w2):
        for r in self.relations:
            if (w1 == r.arg1 and w2 == r.arg2) or (w1 == r.arg2 and w2 == r.arg1):
                return True
        return False

    def get_negative_relations(self):
        word_pos = self.get_sorted_keywords()
        for i in range(1, len(word_pos)):
            w1, _ = word_pos[i - 1]
            w2, _ = word_pos[i]
            if not self.has_relation(w1, w2):
                yield Relation(self, w1, w2, 'None')

    def get_keyword_positions(self, w):
        assert(self.is_keyword(w))
        return get_occur_positions(w, self.sent)

    def merge_keywords(self):
        is_keyword = [False for _ in self.tokens]
        word2label = {}
        to_delete = []
        for keyword_obj in self.keywords:
            word = keyword_obj.word
            if word.find(' ') > 0:
                continue
            word2label[word] = keyword_obj.label
            to_delete.append(keyword_obj)
            ptr = 0
            while True:
                try:
                    pos = self.tokens.index(word, ptr)
                except:
                    break
                is_keyword[pos] = True
                ptr = pos + 1
        for keyword_obj in to_delete:
            self.keywords.remove(keyword_obj)
        ptr = 0
        while ptr < len(self.tokens):
            if not is_keyword[ptr]:
                ptr += 1
                continue
            start = self.token_offset[ptr]
            label_counter = Counter()
            while ptr < len(self.tokens) and is_keyword[ptr]:
                label = word2label[self.tokens[ptr]]
                label_counter[label] += 1
                ptr += 1
            end = self.token_offset[ptr - 1] + len(self.tokens[ptr - 1])
            label = label_counter.most_common(1)[0][0]
            assert(is_valid_label(label))
            self.keywords.add(Keyword(sent_obj=self, word = self.sent[start:end], label=label))

    def __build_tokens_offset(self):
        # build mapping between byte position and tokens
        self.token_offset = []
        ptr = 0
        for token in self.tokens:
            pos = self.sent.index(token, ptr)
            assert(pos >= ptr and pos >= 0)
            self.token_offset.append(pos)
            ptr = pos + len(token)

class Keyword:

    def __init__(self, sent_obj, word, label=None):
        self.sent = sent_obj
        self.word = word
        self.label = label

    def __str__(self):
        return self.word

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

class Relation:

    def __init__(self, sent_obj, arg1, arg2, label='None'):
        # Synonym-of or Hyponym-of
        self.sent = sent_obj
        self.arg1 = arg1
        self.arg2 = arg2
        # Synonym-of is a symmetric relation, do not add both <arg1, arg2> and <arg2, arg1>
        if label == 'Synonym-of' and self.arg1 > self.arg2:
            self.arg1, self.arg2 = self.arg2, self.arg1
        self.label = label

    def __str__(self):
        return '%s_%s_%s' % (self.arg1, self.arg2, self.label)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)
