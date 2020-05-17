
from konlpy.tag import Kkma #pip install jpype1 #pip install konlpy

from konlpy.tag import Twitter

from konlpy.tag import Mecab

from collections import Counter #워드 클라우드에서 단어들을 쉽게 집계하기 위해 사용

from sklearn.feature_extraction.text import TfidfVectorizer #pip install scikit-learn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import normalize

from ole import OleFile #pip install ole-py

from wordcloud import WordCloud


from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models.word2vec import Word2Vec
#from gensim.models.doc2vec import Doc2Vec
import multiprocessing


wikimodel_dir='C:/doc2vec/docmodel/w2v.model'
wikimodel_dir2='C:/doc2vec/model/wiki.model'

dgu_dir='C:/doc2vec/test/Dgu1.txt'
query_dir='C:/doc2vec/query/query.txt'


model = Word2Vec.load(wikimodel_dir2)

import numpy as np

import io
import re
import os
import pytagcloud
import simplejson
import sys

import openpyxl # excel파일 읽기

import olefile # 워드파일읽기

import matplotlib.pyplot as plt #워드클라우드

from google.cloud import vision
from google.cloud.vision import types

from pdfminer.pdfinterp import PDFResourceManager, process_pdf #pip install pdfminer3k
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open #pdf 파일 읽기위함

stopwords = ['번호', '작성자', '작성', '대학', '회수','안내', '공지', '게시','내용','보기','보내기','사항', '학생', '기간', '첨부파일', '학기','신청','학점','학년','지원',
             '국제','방법','확인','선발','지원','모집','교류','실습', '첨부파일','다운로드', '결과']

stopwords2 = []


#i=0


dict_m = []
labels = ['척추동물', '돼지', '수퇘지', '멧돼지', '주둥이', '페커리', '동물', '사파리', '가축']
print(labels)


class SentenceTokenizer(object):

    def __init__(self):

        self.twitter = Twitter()
        self.kkma = Kkma()

    def text2sentences(self, text):
        sentences22 = self.kkma.sentences(text)
        for idx in range(0, len(sentences22)):
            if len(sentences22[idx]) <= 10:
                sentences22[idx - 1] += (' ' + sentences22[idx])
                sentences22[idx] = ''

        return sentences22

    def get_nouns(self, sentences):

        nouns = []

        for sentence in sentences:

            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))

                                       if noun not in stopwords and len(noun) > 1]))

        return nouns


class GraphMatrix(object):

    def __init__(self):
        self.tfidf = TfidfVectorizer()

        self.cnt_vec = CountVectorizer()

        self.graph_sentence = []

    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()

        # print('문장 가공전2222',tfidf_mat)
        # print('크기2222',tfidf_mat.shape)

        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)

        return self.graph_sentence

    # labels= ['척추동물', '돼지', '수퇘지', '멧돼지', '주둥이', '페커리', '동물', '사파리', '가축']

    def build_words_graph(self, sentence):

        cnt_vec_mat2 = self.cnt_vec.fit_transform(sentence).toarray().astype(float)

        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)

        vocab = self.cnt_vec.vocabulary_

        sorted_vocab_idx = sorted(vocab.items())

        print('sorted_vocab_idx', sorted_vocab_idx)

        print(len(sorted_vocab_idx))

        for d, c in sorted_vocab_idx:
            dict_m.append(d)

        for i in labels:
            cnt_col = 0
            for j in dict_m:
                # print('j',j)
                cnt_col += 1
                model_sim = model.wv.similarity(i, j)
                cnt_vec_mat2[:, (cnt_col - 1):cnt_col] += model_sim

        cnt_vec_mat99 = normalize(cnt_vec_mat2.astype(float), axis=0)

        total_cnt = np.dot(cnt_vec_mat99.T, cnt_vec_mat99)

        total_origin = np.dot(cnt_vec_mat.T, cnt_vec_mat)

        print(total_cnt)

        print(total_origin)

        return total_origin, {vocab[word]: word for word in vocab}


class Rank(object):

    def get_ranks(self, graph, d=0.85):  # d = damping factor

        A = graph

        matrix_size = A.shape[0]

        for id in range(matrix_size):

            A[id, id] = 0  # diagonal 부분을 0으로

            link_sum = np.sum(A[:, id])  # A[:, id] = A[:][id]

            if link_sum != 0:
                A[:, id] /= link_sum

            A[:, id] *= -d

            A[id, id] = 1

        # print('A\n',A)
        B = (1 - d) * np.ones((matrix_size, 1))
        # print('B\n',B)

        ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = b

        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):

    def __init__(self, text):

        self.sent_tokenize = SentenceTokenizer()

        # self.sentences = self.sent_tokenize.text2sentences(text)

        self.sentences = text
        # print('\n문장 개수',len(self.sentences))
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)

        # print('nouns\n',nouns)

        self.graph_matrix = GraphMatrix()

        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)

        # print(''self.sent_graph)

        # print('self.sent_graph\n',self.sent_graph)
        # print(self.sent_graph.shape)

        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)

        print('words_graph\n', self.words_graph)
        print(self.words_graph.shape)

        self.rank = Rank()

        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)

        # print('self.sent_rank_idx',self.sent_rank_idx)

        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        # print('self.sorted_sent_rank_idx\n\n',self.sorted_sent_rank_idx)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)

        print('self.word_rank_idx', self.word_rank_idx)

        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

        print('self.sorted_word_rank_idx', self.sorted_word_rank_idx)

    def summarize(self, sent_num=5):

        summary = []

        index = []

        if len(self.sentences) > 30:
            sent_num = 7

        elif len(self.sentences) > 100:
            sent_num = 10

        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)

        index.sort()

        for idx in index:
            summary.append(self.sentences[idx])

        return summary

    def keywords(self, word_num=10):

        rank = Rank()

        rank_idx = rank.get_ranks(self.words_graph)

        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []

        index = []

        # if len(self.sentences) > 30 :
        # word_num=15

        # elif len(self.sentences) > 100:
        # word_num=20

        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        # index.sort()

        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords


def read_pdf_file(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content


path_dir = 'C:/doc2vec/word_pre/txt/'
path_dir2 = 'C:/doc2vec/word_pre/txt2/'
path_dir3 = 'C:/doc2vec/word_pre/txt3/'
path_dir4 = 'C:/doc2vec/word_pre/txt4/'
path_dir5 = 'C:/doc2vec/word_pre/txt5/'
path_dir6 = 'C:/doc2vec/word_pre/txt6/'

jpg_dir = 'C:/doc2vec/word_pre/picture/'
pdf_dir = 'C:/doc2vec/word_pre/pdf/'
hwp_dir = 'C:/doc2vec/word_pre/hwp/'

path_pre = 'C:/doc2vec/result/pre_word/'
summary_dir = 'C:/doc2vec/word_result/summary_test/'
cloud_dir = 'C:/doc2vec/word_pre/cloud/'
keyword_before = 'C:/doc2vec/result/keyword_before/'
full_dir = 'C:/doc2vec/word_result/total_test/'
keyword_after = 'C:/doc2vec/result/keyword_after/'

txt_list = os.listdir(path_dir)  # 폴더안에 있는 파일 리스트로 저장
txt_list2 = os.listdir(path_dir2)
txt_list3 = os.listdir(path_dir3)
txt_list4 = os.listdir(path_dir4)
txt_list5 = os.listdir(path_dir5)
txt_list6 = os.listdir(path_dir6)

txt_pre = os.listdir(path_pre)
str1 = ''
total_word = []
total_word2 = []
t = Twitter()
# dir=path_dir
sys.stdout.flush()

for change in txt_pre:
    total = []  # 통합문서

    remover = change.replace(".txt", "-")
    original = path_pre + change
    etc_pdf = pdf_dir + remover + '0.pdf'
    etc_hwp = hwp_dir + remover + '0.hwp'
    jpg_name = change.replace("txt", "jpg")
    jpg_file = jpg_dir + jpg_name
    png_name = change.replace("txt", "png")
    png_file = jpg_dir + png_name
    pdf_name = change.replace("txt", "pdf")
    pdf_file = pdf_dir + pdf_name
    hwp_name = change.replace("txt", "hwp")
    hwp_file = hwp_dir + hwp_name
    textocr = ""

    print('\n오리지날', original)

    input_file = open(original, "r", encoding="utf-8")
    text1 = input_file.readlines()  # 문서 불러오고 한줄씩 읽기
    total += text1
    input_file.close()

    if (os.path.isfile(jpg_file) == True):
        client = vision.ImageAnnotatorClient()
        file_name = os.path.join(
            os.path.dirname(__file__),
            jpg_file)

        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        textocr = texts[0].description

        jpg1 = ""
        jpg_list = []  # 이미지 리스트
        for x in textocr:
            if (x == '\n'):
                jpg_list.append(jpg1 + '\n')
                jpg1 = ""
            else:
                jpg1 = jpg1 + x
        print('jpg 있음')
        print()
        total += jpg_list

    # print('글 내용',total)

    # total = text1 + jpg_list
    # print('\n전체 내용',total)

    final = []  # 통합문 리스트
    for word in total:

        if (len(word) == 1 or len(word) == 2):
            pass

        else:
            final.append(word)
    # 불필요한 \n 없애기위함

    print('결과 total:\n', final)
    print(len(final))

    total_word = t.nouns(str(final))
    total_word2 += total_word

    # print('원래텍스트 타입', (type(text1)))
    # print('text1:', text1)

    full_name = full_dir + change
    full = open(full_name, 'w', encoding="utf-8")
    full.writelines(final)  # 리스트 형태인 sum2를 저장하기위해 씀

    full.close()

    textrank = TextRank(final)

    sum2 = textrank.summarize()  # 요약문 함수
    keyword_1 = textrank.keywords()
    # newsum=str(sum2)

    temp = summary_dir + change

    f = open(temp, 'w', encoding="utf-8")
    f.writelines(sum2)  # 리스트 형태인 sum2를 저장하기위해 씀
    f.close()

    keyword_temp = keyword_after + change
    keyword_save = open(keyword_temp, 'w', encoding='utf-8')
    for i in keyword_1:
        keyword_save.writelines(i)
        keyword_save.writelines(" ")

    # keyword_save.writelines(keyword_1)
    keyword_save.close()

    # 파일 저장








