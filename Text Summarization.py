import zipfile
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from random import shuffle
from math import log
lemmatizer = WordNetLemmatizer()


zipFile = zipfile.ZipFile("dataset.zip")

ARTICLE = 0
SUMMARY = 1
TITLE = 2
stop_words = set(stopwords.words('english') + list(string.punctuation)) 
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']


def GetPaths(shuff):
    files = {}
    for category in categories: 
        files[category] = []
        for f in zipFile.namelist():
            if 'NewsArticles/' + category in f and '.txt' in f:
                summary = f.replace("NewsArticles", "Summaries")
                files[category].append((f, summary))
    if shuff == False:
        for category in categories:
            files[category].sort()
    else:
        for category in categories:
            shuffle(files[category])
    return files

def GetDataSets(files):
    trainData = {}
    testData = {}
    for category in categories:
        train_no = int(len(files[category]) * 0.75)
        trainData[category] = files[category][:train_no]
        testData[category] = files[category][train_no:]
    return (trainData, testData)


def ReadFile(path):
    f = open(path, "r")
    text = f.read()
    text = text.replace(".", ". ")
    text = text.replace("\n\n", ". ")
    return text


def GetWords(text, stop):
    words = []
    for word in word_tokenize(text):
        if (stop and word not in stop_words and len(word) > 1) or not stop:
            words.append(word)
    return words

def GetBigrams(text, stop, lemma):
    words = GetWords(text, stop)
    if lemma:
        LemmatizeWords(words)
    bigrams = []
    for i in range(0, len(words) - 1):
        bigram = words[i] + ":" + words[i + 1]
        bigrams.append(bigram)
    return bigrams

def Get4grams(text, stop, lemma):
    words = GetWords(text, stop)
    if lemma:
        LemmatizeWords(words)
    fourgrams = []
    for i in range(0, len(words) - 3):
        fourgram = words[i] + ":" + words[i + 1] + ":" + words[i + 2] + ":" + words[i + 3]
        fourgrams.append(fourgram)
    return fourgrams

def LemmatizeWords(words):
    for i in range(0, len(words)):
        words[i] = lemmatizer.lemmatize(words[i])
    return

def GetSentences(trainData):
    sentences = {}
    sentences_no = {}
    for category in categories:
        sentences[category] = []
        sentences_no[category] = (0, 0)
        for path in trainData[category]:
            article_text = ReadFile(path[ARTICLE])
            summary_text = ReadFile(path[SUMMARY])

            summary_sent = sent_tokenize(summary_text)
            article_sent = []
            isTitle = True
            title = ""
            for sent in sent_tokenize(article_text):
                if isTitle:
                    title = sent
                    isTitle = False
                else:
                    if sent not in summary_sent:
                        article_sent.append(sent)
            sentences[category].append((article_sent, summary_sent, title))
            sentences_no[category] = (sentences_no[category][0] + len(article_sent), sentences_no[category][1] + len(summary_sent))
    return (sentences, sentences_no)

def evaluate(testData, results, category, prediction_func, format_html):
    total_common_words = 0
    total_reference_words = 0
    total_predicted_words = 0
    total_common_bigrams = 0
    total_reference_bigrams = 0
    total_predicted_bigrams = 0
    total_common_4grams = 0
    total_reference_4grams = 0
    total_predicted_4grams = 0
    for path in testData[category]:
        predicted_summary = prediction_func(results, category, path)
        reference_summary = ReadFile(path[SUMMARY])
        
        #calcul BLEU1, ROUGE1
        predicted_words = GetWords(predicted_summary, False)
        reference_words = GetWords(reference_summary, False)

        reference_word_count = {}
        predicted_word_count = {}
        for word in reference_words:
            if word not in reference_word_count:
                reference_word_count[word] = 1
            else:
                reference_word_count[word] += 1
        for word in predicted_words:
            if word not in predicted_word_count:
                predicted_word_count[word] = 1
            else:
                predicted_word_count[word] += 1
            if word in reference_word_count:
                if reference_word_count[word] >= predicted_word_count[word]:
                    total_common_words += 1

        total_reference_words += len(reference_words)
        total_predicted_words += len(predicted_words)

        #calcul BLEU2, ROUGE2
        predicted_bigrams = GetBigrams(predicted_summary, False, False)
        reference_bigrams = GetBigrams(reference_summary, False, False)

        reference_bigram_count = {}
        predicted_bigram_count = {}
        for bigram in reference_bigrams:
            if bigram not in reference_bigram_count:
                reference_bigram_count[bigram] = 1
            else:
                reference_bigram_count[bigram] += 1
        for bigram in predicted_bigrams:
            if bigram not in predicted_bigram_count:
                predicted_bigram_count[bigram] = 1
            else:
                predicted_bigram_count[bigram] += 1
            if bigram in reference_bigram_count:
                if reference_bigram_count[bigram] >= predicted_bigram_count[bigram]:
                    total_common_bigrams += 1

        total_reference_bigrams += len(reference_bigrams)
        total_predicted_bigrams += len(predicted_bigrams)

        #calcul BLEU4, ROUGE4
        predicted_4grams = Get4grams(predicted_summary, False, False)
        reference_4grams = Get4grams(reference_summary, False, False)

        reference_4gram_count = {}
        predicted_4gram_count = {}
        for fourgram in reference_4grams:
            if fourgram not in reference_4gram_count:
                reference_4gram_count[fourgram] = 1
            else:
                reference_4gram_count[fourgram] += 1
        for fourgram in predicted_4grams:
            if fourgram not in predicted_4gram_count:
                predicted_4gram_count[fourgram] = 1
            else:
                predicted_4gram_count[fourgram] += 1
            if fourgram in reference_4gram_count:
                if reference_4gram_count[fourgram] >= predicted_4gram_count[fourgram]:
                    total_common_4grams += 1
        total_reference_4grams += len(reference_4grams)
        total_predicted_4grams += len(predicted_4grams)

    BLEU1 = total_common_words / total_reference_words
    ROUGE1 = total_common_words / total_predicted_words

    BLEU2 = total_common_bigrams / total_reference_bigrams
    ROUGE2 = total_common_bigrams / total_predicted_bigrams

    BLEU4 = total_common_4grams / total_reference_4grams
    ROUGE4 = total_common_4grams / total_predicted_4grams

    if format_html:
        print("<td>" +  str(round(BLEU1, 3)) + "</td>")
        print("<td>" +  str(round(ROUGE1, 3)) + "</td>")
        print("<td>" +  str(round(BLEU2, 3)) + "</td>")
        print("<td>" +  str(round(ROUGE2, 3)) + "</td>")
        print("<td>" +  str(round(BLEU4, 3)) + "</td>")
        print("<td>" +  str(round(ROUGE4, 3)) + "</td>")
    else:
        print("*Category: " + category)
        print("--------------------------------")
        print("BLEU1: " + str(BLEU1))
        print("ROUGE1: " + str(ROUGE1))
        print("BLEU2: " + str(BLEU2))
        print("ROUGE2: " + str(ROUGE2))
        print("BLEU4: " + str(BLEU4))
        print("ROUGE4: " + str(ROUGE4))
        print("--------------------------------")
        print(" ")

def GetAvgSummarySentPerc(trainData):
    avg = {}
    for category in categories:
        article_sent_nr = 0
        summary_sent_nr = 0
        avg[category] = 0
        for path in trainData[category]:
            article_text = ReadFile(path[0])
            summary_text = ReadFile(path[1])
            article_sent_nr += len(sent_tokenize(article_text))
            summary_sent_nr += len(sent_tokenize(summary_text))
        avg[category] = summary_sent_nr / article_sent_nr
    return avg

#---------------------------------------------------------------------- NAIVE BAYES - UNIGRAME

def CreateWordVocabulary(sentences, stop, lemma):
    vocabulary = {}
    words_no = {}
    for category in categories:
        vocabulary[category] = {}
        words_no[category] = (0, 0)
        for doc in sentences[category]:
            for sentence in doc[ARTICLE]:
                sent_words = GetWords(sentence, stop)
                if lemma:
                    LemmatizeWords(sent_words)
                for word in sent_words:
                    words_no[category] = (words_no[category][ARTICLE] + 1, words_no[category][SUMMARY])
                    if word not in vocabulary[category]:
                        vocabulary[category][word] = (1, 0)
                    else:
                        vocabulary[category][word] = (vocabulary[category][word][ARTICLE] + 1, vocabulary[category][word][SUMMARY])
            for sentence in doc[SUMMARY]:
                sent_words = GetWords(sentence, stop)
                if lemma:
                    LemmatizeWords(sent_words)
                for word in sent_words:
                    words_no[category] = (words_no[category][ARTICLE], words_no[category][SUMMARY] + 1)
                    if word not in vocabulary[category]:
                        vocabulary[category][word] = (0, 1)
                    else:
                        vocabulary[category][word] = (vocabulary[category][word][ARTICLE], vocabulary[category][word][SUMMARY] + 1)
    return (vocabulary, words_no)
                
def PredictSummary_NB_Unigrams(trainingResults, category, path, alpha = 1):
    predicted_sentences = []

    (vocabulary, words_no, sentences_no, stop, lemma) = trainingResults
    article_text = ReadFile(path[ARTICLE])

    article_sent = []
    isTitle = True
    for sent in sent_tokenize(article_text):
        if not isTitle:
            article_sent.append(sent)
        else:
            isTitle = False

    for sentence in article_sent:
        sent_words = GetWords(sentence, stop)
        if lemma:
            LemmatizeWords(sent_words)
        log_summary = log(sentences_no[category][SUMMARY] / (sentences_no[category][SUMMARY] + sentences_no[category][ARTICLE]))
        log_article = log(sentences_no[category][ARTICLE] / (sentences_no[category][ARTICLE] + sentences_no[category][SUMMARY]))
        for word in sent_words:
            if word in vocabulary[category]:
                log_article = log_article + log((vocabulary[category][word][ARTICLE] + alpha) / (words_no[category][ARTICLE] + len(vocabulary[category].keys()) * alpha))
                log_summary = log_summary + log((vocabulary[category][word][SUMMARY] + alpha) / (words_no[category][SUMMARY] + len(vocabulary[category].keys()) * alpha))
            else:
                log_article = log_article + log(alpha / (words_no[category][ARTICLE] + len(vocabulary[category].keys()) * alpha))
                log_summary = log_summary + log(alpha / (words_no[category][SUMMARY] + len(vocabulary[category].keys()) * alpha))
        if log_summary > log_article:
            predicted_sentences.append(sentence)
    predicted_summary = ""
    for predicted_sentence in predicted_sentences:
        predicted_summary = predicted_summary + " " + predicted_sentence
    return predicted_summary

def NaiveBayesWords(stop, lemma, shuffle, format_html):
    files = GetPaths(shuffle)
    (trainData, testData) = GetDataSets(files)
    (sentences, sentences_no) = GetSentences(trainData)
    (vocabulary, words_no) = CreateWordVocabulary(sentences, stop, lemma)
    trainingResults = (vocabulary, words_no, sentences_no, stop, lemma)
    for category in categories:
        if format_html:
            print("<tr>")
            print("<td>" +  "unigrame" + "</td>") # n-grame
            print("<td>" +  category + "</td>") # categorie
            if stop: #stop words
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
            if lemma: # lematizare
                print("<td>" +  "DA" + "</td>")
            else: 
                print("<td>" +  "NU" + "</td>")
        evaluate(testData, trainingResults, category, PredictSummary_NB_Unigrams, format_html)
        if format_html:
            print("</tr>")
    #predicted_summary = PredictSummary_NB_Unigrams(trainingResults, 'tech', testData['tech'][20])
    #print(predicted_summary)

#---------------------------------------------------------------------- NAIVE BAYES - BIGRAME

def CreateBigramVocabulary(sentences, stop, lemma):
    vocabulary = {}
    bigrams_no = {}
    for category in categories:
        vocabulary[category] = {}
        bigrams_no[category] = (0, 0)
        for doc in sentences[category]:
            for sentence in doc[ARTICLE]:
                sent_bigrams = GetBigrams(sentence, stop, lemma)
                for bigram in sent_bigrams:
                    bigrams_no[category] = (bigrams_no[category][ARTICLE] + 1, bigrams_no[category][SUMMARY])
                    if bigram not in vocabulary[category]:
                        vocabulary[category][bigram] = (1, 0)
                    else:
                        vocabulary[category][bigram] = (vocabulary[category][bigram][ARTICLE] + 1, vocabulary[category][bigram][SUMMARY])
            for sentence in doc[SUMMARY]:
                sent_bigrams = GetBigrams(sentence, stop, lemma)
                for bigram in sent_bigrams:
                    bigrams_no[category] = (bigrams_no[category][ARTICLE], bigrams_no[category][SUMMARY] + 1)
                    if bigram not in vocabulary[category]:
                        vocabulary[category][bigram] = (0, 1)
                    else:
                        vocabulary[category][bigram] = (vocabulary[category][bigram][ARTICLE], vocabulary[category][bigram][SUMMARY] + 1)
    return (vocabulary, bigrams_no)

def PredictSummary_NB_Bigrams(trainingResults, category, path, alpha = 1):
    predicted_sentences = []

    (vocabulary, bigrams_no, sentences_no, stop, lemma) = trainingResults
    article_text = ReadFile(path[ARTICLE])

    article_sent = []
    isTitle = True
    for sent in sent_tokenize(article_text):
        if not isTitle:
            article_sent.append(sent)
        else:
            isTitle = False

    for sentence in article_sent:
        sent_bigrams = GetBigrams(sentence, stop, lemma)
        log_summary = log(sentences_no[category][SUMMARY] / (sentences_no[category][SUMMARY] + sentences_no[category][ARTICLE]))
        log_article = log(sentences_no[category][ARTICLE] / (sentences_no[category][ARTICLE] + sentences_no[category][SUMMARY]))
        for bigram in sent_bigrams:
            if bigram in vocabulary[category]:
                log_article = log_article + log((vocabulary[category][bigram][ARTICLE] + alpha) / (bigrams_no[category][ARTICLE] + len(vocabulary[category].keys()) * alpha))
                log_summary = log_summary + log((vocabulary[category][bigram][SUMMARY] + alpha) / (bigrams_no[category][SUMMARY] + len(vocabulary[category].keys()) * alpha))
            else:
                log_article = log_article + log(alpha / (bigrams_no[category][ARTICLE] + len(vocabulary[category].keys()) * alpha))
                log_summary = log_summary + log(alpha / (bigrams_no[category][SUMMARY] + len(vocabulary[category].keys()) * alpha))
        if log_summary > log_article:
            predicted_sentences.append(sentence)
    predicted_summary = ""
    for predicted_sentence in predicted_sentences:
        predicted_summary = predicted_summary + " " + predicted_sentence
    return predicted_summary

def NaiveBayesBigrams(stop, lemma, shuffle, format_html):
    files = GetPaths(shuffle)
    (trainData, testData) = GetDataSets(files)
    (sentences, sentences_no) = GetSentences(trainData)
    (vocabulary, bigrams_no) = CreateBigramVocabulary(sentences, stop, lemma)
    trainingResults = (vocabulary, bigrams_no, sentences_no, stop, lemma)
    for category in categories:
        if format_html:
            print("<tr>")
            print("<td>" +  "bigrame" + "</td>") # n-grame
            print("<td>" +  category + "</td>") # categorie
            if stop: #stop words
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
            if lemma: # lematizare
                print("<td>" +  "DA" + "</td>")
            else: 
                print("<td>" +  "NU" + "</td>")
        evaluate(testData, trainingResults, category, PredictSummary_NB_Bigrams, format_html)
        if format_html:
            print("</tr>")

    #predicted_summary = PredictSummary_NB_Bigrams(trainingResults, 'tech', testData['tech'][20])
    #print(predicted_summary)

#---------------------------------------------------------------------- NAIVE BAYES - 4-GRAME

def Create4gramVocabulary(sentences, stop, lemma):
    vocabulary = {}
    fourgrams_no = {}
    for category in categories:
        vocabulary[category] = {}
        fourgrams_no[category] = (0, 0)
        for doc in sentences[category]:
            for sentence in doc[ARTICLE]:
                sent_4grams = Get4grams(sentence, stop, lemma)
                for fourgram in sent_4grams:
                    fourgrams_no[category] = (fourgrams_no[category][ARTICLE] + 1, fourgrams_no[category][SUMMARY])
                    if fourgram not in vocabulary[category]:
                        vocabulary[category][fourgram] = (1, 0)
                    else:
                        vocabulary[category][fourgram] = (vocabulary[category][fourgram][ARTICLE] + 1, vocabulary[category][fourgram][SUMMARY])
            for sentence in doc[SUMMARY]:
                sent_4grams = Get4grams(sentence, stop, lemma)
                for fourgram in sent_4grams:
                    fourgrams_no[category] = (fourgrams_no[category][ARTICLE], fourgrams_no[category][SUMMARY] + 1)
                    if fourgram not in vocabulary[category]:
                        vocabulary[category][fourgram] = (0, 1)
                    else:
                        vocabulary[category][fourgram] = (vocabulary[category][fourgram][ARTICLE], vocabulary[category][fourgram][SUMMARY] + 1)
    return (vocabulary, fourgrams_no)

def PredictSummary_NB_4grams(trainingResults, category, path, alpha = 1):
    predicted_sentences = []

    (vocabulary, fourgrams_no, sentences_no, stop, lemma) = trainingResults
    article_text = ReadFile(path[ARTICLE])

    article_sent = []
    isTitle = True
    for sent in sent_tokenize(article_text):
        if not isTitle:
            article_sent.append(sent)
        else:
            isTitle = False

    for sentence in article_sent:
        sent_4grams = Get4grams(sentence, stop, lemma)
        log_summary = log(sentences_no[category][SUMMARY] / (sentences_no[category][SUMMARY] + sentences_no[category][ARTICLE]))
        log_article = log(sentences_no[category][ARTICLE] / (sentences_no[category][ARTICLE] + sentences_no[category][SUMMARY]))
        for fourgram in sent_4grams:
            if fourgram in vocabulary[category]:
                log_article = log_article + log((vocabulary[category][fourgram][ARTICLE] + alpha) / (fourgrams_no[category][ARTICLE] + len(vocabulary[category].keys()) * alpha))
                log_summary = log_summary + log((vocabulary[category][fourgram][SUMMARY] + alpha) / (fourgrams_no[category][SUMMARY] + len(vocabulary[category].keys()) * alpha))
            else:
                log_article = log_article + log(alpha / (fourgrams_no[category][ARTICLE] + len(vocabulary[category].keys()) * alpha))
                log_summary = log_summary + log(alpha / (fourgrams_no[category][SUMMARY] + len(vocabulary[category].keys()) * alpha))
        if log_summary > log_article:
            predicted_sentences.append(sentence)
    predicted_summary = ""
    for predicted_sentence in predicted_sentences:
        predicted_summary = predicted_summary + " " + predicted_sentence
    return predicted_summary

def NaiveBayes4grams(stop, lemma, shuffle, format_html):
    files = GetPaths(shuffle)
    (trainData, testData) = GetDataSets(files)
    (sentences, sentences_no) = GetSentences(trainData)
    (vocabulary, fourgrams_no) = Create4gramVocabulary(sentences, stop, lemma)
    trainingResults = (vocabulary, fourgrams_no, sentences_no, stop, lemma)
    
    for category in categories:
        if format_html:
            print("<tr>")
            print("<td>" +  "4-grame" + "</td>") # n-grame
            print("<td>" +  category + "</td>") # categorie
            if stop: #stop words
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
            if lemma: # lematizare
                print("<td>" +  "DA" + "</td>")
            else: 
                print("<td>" +  "NU" + "</td>")
        evaluate(testData, trainingResults, category, PredictSummary_NB_4grams, format_html)
        if format_html:
            print("</tr>")
    
    #predicted_summary = PredictSummary_NB_4grams(trainingResults, 'entertainment', testData['entertainment'][50])
    #print("---------------------------Referinta-------------------------------------")
    #print(ReadFile(testData['entertainment'][50][SUMMARY]))
    #print("---------------------------NAIVE BAYES-----------------------------------")
    #print(predicted_summary)

#---------------------------------------------------------------------- TF-IDF - UNIGRAME

def GetWordsIDF(trainData, stop, lemma, nouns):
    idf_vocabulary = {}
    for category in categories:
        idf_vocabulary[category] = {}
        for path in trainData[category]:
            current_vocab = {}
            article_text = ReadFile(path[ARTICLE])
            words = GetWords(article_text, stop)
            if lemma:
                LemmatizeWords(words)
            if nouns:
                pos_tagged = nltk.pos_tag(words)
                words = []
                for word in pos_tagged:
                    if word[1] == 'NN':
                        words.append(word[0])
            for word in words:
                if word not in current_vocab:
                    if word in idf_vocabulary[category]:
                        idf_vocabulary[category][word] += 1
                    else:
                        idf_vocabulary[category][word] = 1
                current_vocab[word] = 1
    for category in categories:
        for word in idf_vocabulary[category].keys():
            idf_vocabulary[category][word] = log(len(trainData[category]) / idf_vocabulary[category][word]) / log(len(trainData[category]))
    return idf_vocabulary

def Word_TF_IDF_Score(words, idf_vocabulary):
    tf_vocabulary = {}
    for word in words:
        if word not in tf_vocabulary:
            tf_vocabulary[word] = 1
        else:
            tf_vocabulary[word] += 1
    tfidf_vocab = {}
    max_score = 0
    for word in tf_vocabulary.keys():
        if word not in idf_vocabulary:
            tfidf_vocab[word] = tf_vocabulary[word]
        else:
            tfidf_vocab[word] = tf_vocabulary[word] * idf_vocabulary[word]
        if tfidf_vocab[word] > max_score:
            max_score = tfidf_vocab[word]
    for word in tfidf_vocab.keys():
        tfidf_vocab[word] = tfidf_vocab[word] / max_score
    return tfidf_vocab

def PredictSummary_TFIDF_Words(trainingResults, category, path, alpha = 1):

    (idf_vocabulary, stop, lemma, nouns, title_const, doc_location, avg_summary_sent_perc) = trainingResults
    article_text = ReadFile(path[ARTICLE])
    
    words = GetWords(article_text, stop)
    if lemma:
        LemmatizeWords(words)
    if nouns:
        pos_tagged = nltk.pos_tag(words)
        words = []
        for word in pos_tagged:
            if word[1] == 'NN':
                words.append(word[0])

    tfidf_vocab = Word_TF_IDF_Score(words, idf_vocabulary[category])

    article_sent = []
    isTitle = True
    for sent in sent_tokenize(article_text):
        if not isTitle:
            article_sent.append(sent)
        else:
            isTitle = False
            title = sent
    
    if title_const != 0:
        title_words = GetWords(title, stop)
        if lemma:
            LemmatizeWords(title_words)

    sentences_scores = []
    cnt = 1
    for sentence in article_sent:
        sent_words = GetWords(sentence, stop)
        if lemma:
            LemmatizeWords(sent_words)

        sentence_score = 0
        for word in sent_words:
            if word in tfidf_vocab:
                sentence_score += (tfidf_vocab[word] / len(sent_words))
        if title_const != 0:
            common_words = 0
            for tword in title_words:
                if tword in sent_words:
                    common_words += 1
            title_score = title_const * (common_words / len(title_words))
            sentence_score += title_score
        if doc_location:
            sentence_score = sentence_score * (cnt / len(article_sent))
        cnt = cnt + 1
        sentences_scores.append((sentence, sentence_score))

    for i in range(0, len(sentences_scores)):
        for j in range(i + 1, len(sentences_scores)):
            if sentences_scores[i][1] < sentences_scores[j][1]:
                aux = sentences_scores[i]
                sentences_scores[i] = sentences_scores[j]
                sentences_scores[j] = aux

    predicted_sentences = []
    k = avg_summary_sent_perc[category] * len(sentences_scores)
    k = int(k)
    for i in range(0, k):
        predicted_sentences.append(sentences_scores[i][0])

    predicted_summary = ""
    for predicted_sentence in predicted_sentences:
        predicted_summary = predicted_summary + " " + predicted_sentence
    
    return predicted_summary

def TF_IDF_Words(stop, lemma, nouns, doc_location, title_const, shuffle, format_html):
    files = GetPaths(shuffle)
    (trainData, testData) = GetDataSets(files)
    idf_vocabulary = GetWordsIDF(trainData, stop, lemma, nouns)
    avg_summary_sent_perc = GetAvgSummarySentPerc(trainData)
    trainingResults = (idf_vocabulary, stop, lemma, nouns, title_const, doc_location, avg_summary_sent_perc)
    #predicted_summary = PredictSummary_TFIDF_Words(trainingResults, 'entertainment', testData['entertainment'][50])
    #print("---------------------------TFIDF WORDS-----------------------------------")
    #print(predicted_summary)
    
    for category in categories:
        if format_html:
            print("<tr>")
            print("<td>" +  "unigrame" + "</td>") # n-grame
            print("<td>" +  category + "</td>") # categorie
            if nouns: #doar substantive
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
            if title_const > 0: # pondere titlu
                print("<td>" +  "DA" + "</td>")
            else: 
                print("<td>" +  "NU" + "</td>")
            print("<td>" +  str(title_const) + "</td>")
            if doc_location: # pondere locatie document
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
        evaluate(testData, trainingResults, category, PredictSummary_TFIDF_Words, format_html)
        if format_html:
            print("</tr>")
            


#---------------------------------------------------------------------- TF-IDF - BIGRAME

def Bigram_TF_IDF_Score(bigrams, idf_vocabulary):
    tf_vocabulary = {}
    for bigram in bigrams:
        if bigram not in tf_vocabulary:
            tf_vocabulary[bigram] = 1
        else:
            tf_vocabulary[bigram] += 1
    tfidf_vocab = {}
    max_score = 0
    for bigram in tf_vocabulary.keys():
        if bigram not in idf_vocabulary:
            tfidf_vocab[bigram] = tf_vocabulary[bigram]
        else:
            tfidf_vocab[bigram] = tf_vocabulary[bigram] * idf_vocabulary[bigram]
        if tfidf_vocab[bigram] > max_score:
            max_score = tfidf_vocab[bigram]
    for bigram in tfidf_vocab.keys():
        tfidf_vocab[bigram] = tfidf_vocab[bigram] / max_score
    return tfidf_vocab

def GetBigramsIDF(trainData, stop, lemma, nouns):
    idf_vocabulary = {}
    for category in categories:
        idf_vocabulary[category] = {}
        for path in trainData[category]:
            vocab = {}
            article_text = ReadFile(path[ARTICLE])
            words = GetWords(article_text, stop)
            if lemma:
                LemmatizeWords(words)
            bigrams = []
            if nouns:
                pos_tagged = nltk.pos_tag(words)
                for i in range(0, len(pos_tagged) - 1):
                    if pos_tagged[i][1] == 'NN' or pos_tagged[i + 1][1] == 'NN':
                        bigram = pos_tagged[i][0] + ":" + pos_tagged[i + 1][0]
                        bigrams.append(bigram)
            else:
                for i in range(0, len(words) - 1):
                    bigram = words[i] + ":" + words[i + 1]
                    bigrams.append(bigram)

            for bigram in bigrams:
                if bigram not in vocab:
                    if bigram in idf_vocabulary[category]:
                        idf_vocabulary[category][bigram] += 1
                    else:
                        idf_vocabulary[category][bigram] = 1
                vocab[bigram] = 1

    for category in categories:
        for bigram in idf_vocabulary[category].keys():
            idf_vocabulary[category][bigram] = log(len(trainData[category]) / idf_vocabulary[category][bigram]) / log(len(trainData[category]))
    return idf_vocabulary

def PredictSummary_TFIDF_Bigrams(trainingResults, category, path, alpha = 1):

    (idf_vocabulary, stop, lemma, nouns, title_const, doc_location, avg_summary_sent_perc) = trainingResults
    article_text = ReadFile(path[ARTICLE])

    words = GetWords(article_text, stop)
    if lemma:
        LemmatizeWords(words)

    bigrams = []
    if nouns:
        pos_tagged = nltk.pos_tag(words)
        for i in range(0, len(pos_tagged) - 1):
            if pos_tagged[i][1] == 'NN' or pos_tagged[i + 1][1] == 'NN':
                bigram = pos_tagged[i][0] + ":" + pos_tagged[i + 1][0]
                bigrams.append(bigram)
    else:
        for i in range(0, len(words) - 1):
            bigram = words[i] + ":" + words[i + 1]
            bigrams.append(bigram)

    tfidf_vocab = Bigram_TF_IDF_Score(bigrams, idf_vocabulary[category])

    article_sent = []
    isTitle = True
    for sent in sent_tokenize(article_text):
        if not isTitle:
            article_sent.append(sent)
        else:
            isTitle = False
            title = sent
    
    if title_const != 0: 
        title_words = GetWords(title, stop)
        if lemma:
            LemmatizeWords(title_words)
        title_bigrams = []
        if nouns:
            title_pos_tagged = nltk.pos_tag(title_words)
            for i in range(0, len(title_pos_tagged) - 1):
                if title_pos_tagged[i][1] == 'NN' or title_pos_tagged[i + 1][1] == 'NN':
                    bigram = title_pos_tagged[i][0] + ":" + title_pos_tagged[i + 1][0]
                    title_bigrams.append(bigram)
        else:
            for i in range(0, len(title_words) - 1):
                bigram = title_words[i] + ":" + title_words[i + 1]
                title_bigrams.append(bigram)

    sentences_scores = []
    cnt = 1
    for sentence in article_sent:
        sent_words = GetWords(sentence, stop)
        if lemma:
            LemmatizeWords(sent_words)
        sent_bigrams = []
        if nouns:
            sent_pos_tagged = nltk.pos_tag(sent_words)
            for i in range(0, len(sent_pos_tagged) - 1):
                if sent_pos_tagged[i][1] == 'NN' or sent_pos_tagged[i + 1][1] == 'NN':
                    bigram = sent_pos_tagged[i][0] + ":" + sent_pos_tagged[i + 1][0]
                    sent_bigrams.append(bigram)
        else:
            for i in range(0, len(sent_words) - 1):
                bigram = sent_words[i] + ":" + sent_words[i + 1]
                sent_bigrams.append(bigram)

        sentence_score = 0
        for bigram in sent_bigrams:
            if bigram in tfidf_vocab:
                sentence_score += (tfidf_vocab[bigram] / len(sent_bigrams))
        if title_const != 0:
            common_bigrams = 0
            for tbigram in title_bigrams:
                if tbigram in sent_bigrams:
                    common_bigrams += 1
            if len(title_bigrams) > 0:
                title_score = title_const * (common_bigrams / len(title_bigrams))
            else:
                title_score = 0
            sentence_score += title_score
        if doc_location:
            sentence_score = sentence_score * (cnt / len(article_sent))
        cnt = cnt + 1
        sentences_scores.append((sentence, sentence_score))

    for i in range(0, len(sentences_scores)):
        for j in range(i + 1, len(sentences_scores)):
            if sentences_scores[i][1] < sentences_scores[j][1]:
                aux = sentences_scores[i]
                sentences_scores[i] = sentences_scores[j]
                sentences_scores[j] = aux

    predicted_sentences = []
    k = avg_summary_sent_perc[category] * len(sentences_scores)
    k = int(k)
    for i in range(0, k):
        predicted_sentences.append(sentences_scores[i][0])

    predicted_summary = ""
    for predicted_sentence in predicted_sentences:
        predicted_summary = predicted_summary + " " + predicted_sentence
    
    return predicted_summary

def TF_IDF_Bigrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html):
    files = GetPaths(shuffle)
    (trainData, testData) = GetDataSets(files)
    idf_vocabulary = GetBigramsIDF(trainData, stop, lemma, nouns)
    avg_summary_sent_perc = GetAvgSummarySentPerc(trainData)
    trainingResults = (idf_vocabulary, stop, lemma, nouns, title_const, doc_location, avg_summary_sent_perc)
    for category in categories:
        if format_html:
            print("<tr>")
            print("<td>" +  "bigrame" + "</td>") # n-grame
            print("<td>" +  category + "</td>") # categorie
            if nouns: #doar substantive
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
            if title_const > 0: # pondere titlu
                print("<td>" +  "DA" + "</td>")
            else: 
                print("<td>" +  "NU" + "</td>")
            print("<td>" +  str(title_const) + "</td>")
            if doc_location: # pondere locatie document
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
        evaluate(testData, trainingResults, category, PredictSummary_TFIDF_Bigrams, format_html)
        if format_html:
            print("</tr>")



#---------------------------------------------------------------------- TF-IDF - 4-GRAME

def Fourgram_TF_IDF_Score(fourgrams, idf_vocabulary):
    tf_vocabulary = {}
    for fourgram in fourgrams:
        if fourgram not in tf_vocabulary:
            tf_vocabulary[fourgram] = 1
        else:
            tf_vocabulary[fourgram] += 1
    tfidf_vocab = {}
    max_score = 0
    for fourgram in tf_vocabulary.keys():
        if fourgram not in idf_vocabulary:
            tfidf_vocab[fourgram] = tf_vocabulary[fourgram]
        else:
            tfidf_vocab[fourgram] = tf_vocabulary[fourgram] * idf_vocabulary[fourgram]
        if tfidf_vocab[fourgram] > max_score:
            max_score = tfidf_vocab[fourgram]
    for fourgram in tfidf_vocab.keys():
        tfidf_vocab[fourgram] = tfidf_vocab[fourgram] / max_score
    return tfidf_vocab

def GetFourgramsIDF(trainData, stop, lemma, nouns):
    idf_vocabulary = {}
    for category in categories:
        idf_vocabulary[category] = {}
        for path in trainData[category]:
            vocab = {}
            article_text = ReadFile(path[ARTICLE])
            words = GetWords(article_text, stop)
            if lemma:
                LemmatizeWords(words)
            fourgrams = []
            if nouns:
                pos_tagged = nltk.pos_tag(words)
                for i in range(0, len(pos_tagged) - 3):
                    if pos_tagged[i][1] == 'NN' or pos_tagged[i + 1][1] == 'NN' or pos_tagged[i + 2][1] == 'NN' or pos_tagged[i + 3][1] == 'NN':
                        fourgram = pos_tagged[i][0] + ":" + pos_tagged[i + 1][0] + ":" + pos_tagged[i + 2][0] + ":" + pos_tagged[i + 3][0]
                        fourgrams.append(fourgram)
            else:
                for i in range(0, len(words) - 3):
                    fourgram = words[i] + ":" + words[i + 1] + ":" + words[i + 2] + ":" + words[i + 3]
                    fourgrams.append(fourgram)

            for fourgram in fourgrams:
                if fourgram not in vocab:
                    if fourgram in idf_vocabulary[category]:
                        idf_vocabulary[category][fourgram] += 1
                    else:
                        idf_vocabulary[category][fourgram] = 1
                vocab[fourgram] = 1

    for category in categories:
        for fourgram in idf_vocabulary[category].keys():
            idf_vocabulary[category][fourgram] = log(len(trainData[category]) / idf_vocabulary[category][fourgram]) / log(len(trainData[category])) 
    return idf_vocabulary

def PredictSummary_TFIDF_Fourgrams(trainingResults, category, path, alpha = 1):

    (idf_vocabulary, stop, lemma, nouns, title_const, doc_location, avg_summary_sent_perc) = trainingResults
    article_text = ReadFile(path[ARTICLE])

    words = GetWords(article_text, stop)
    if lemma:
        LemmatizeWords(words)

    fourgrams = []
    if nouns:
        pos_tagged = nltk.pos_tag(words)
        for i in range(0, len(pos_tagged) - 3):
            if pos_tagged[i][1] == 'NN' or pos_tagged[i + 1][1] == 'NN' or pos_tagged[i + 2][1] == 'NN' or pos_tagged[i + 3][1] == 'NN':
                fourgram = pos_tagged[i][0] + ":" + pos_tagged[i + 1][0] + ":" + pos_tagged[i + 2][0] + ":" + pos_tagged[i + 3][0]
                fourgrams.append(fourgram)
    else:
        for i in range(0, len(words) - 3):
            fourgram = words[i] + ":" + words[i + 1] + ":" + words[i + 2] + ":" + words[i + 3]
            fourgrams.append(fourgram)

    tfidf_vocab = Fourgram_TF_IDF_Score(fourgrams, idf_vocabulary[category])

    article_sent = []
    isTitle = True
    for sent in sent_tokenize(article_text):
        if not isTitle:
            article_sent.append(sent)
        else:
            isTitle = False
            title = sent
    
    if title_const != 0: 
        title_words = GetWords(title, stop)
        if lemma:
            LemmatizeWords(title_words)
        title_fourgrams = []
        if nouns:
            title_pos_tagged = nltk.pos_tag(title_words)
            for i in range(0, len(title_pos_tagged) - 3):
                if title_pos_tagged[i][1] == 'NN' or title_pos_tagged[i + 1][1] == 'NN' or title_pos_tagged[i + 2][1] == 'NN' or title_pos_tagged[i + 3][1] == 'NN':
                    fourgram = title_pos_tagged[i][0] + ":" + title_pos_tagged[i + 1][0] + ":" + title_pos_tagged[i + 2][0] + ":" + title_pos_tagged[i + 3][0]
                    title_fourgrams.append(fourgram)
        else:
            for i in range(0, len(title_words) - 3):
                fourgram = title_words[i] + ":" + title_words[i + 1] + ":" + title_words[i + 2] + ":" + title_words[i + 3]
                title_fourgrams.append(fourgram)

    sentences_scores = []
    cnt = 1
    for sentence in article_sent:
        sent_words = GetWords(sentence, stop)
        if lemma:
            LemmatizeWords(sent_words)
        sent_fourgrams = []
        if nouns:
            sent_pos_tagged = nltk.pos_tag(sent_words)
            for i in range(0, len(sent_pos_tagged) - 3):
                if sent_pos_tagged[i][1] == 'NN' or sent_pos_tagged[i + 1][1] == 'NN' or sent_pos_tagged[i + 2][1] == 'NN' or sent_pos_tagged[i + 3][1] == 'NN':
                    fourgram = sent_pos_tagged[i][0] + ":" + sent_pos_tagged[i + 1][0] + ":" + sent_pos_tagged[i + 2][0] + ":" + sent_pos_tagged[i + 3][0]
                    sent_fourgrams.append(fourgram)
        else:
            for i in range(0, len(sent_words) - 3):
                fourgram = sent_words[i] + ":" + sent_words[i + 1]
                sent_fourgrams.append(fourgram)

        sentence_score = 0
        for fourgram in sent_fourgrams:
            if fourgram in tfidf_vocab:
                sentence_score += (tfidf_vocab[fourgram] / len(sent_fourgrams))
        if title_const != 0:
            common_fourgrams = 0
            for tfourgram in title_fourgrams:
                if tfourgram in sent_fourgrams:
                    common_fourgrams += 1
            if len(title_fourgrams) > 0:
                title_score = title_const * (common_fourgrams / len(title_fourgrams))
            else:
                title_score = 0
            sentence_score += title_score
        if doc_location:
            sentence_score = sentence_score * (cnt / len(article_sent))
        cnt = cnt + 1
        sentences_scores.append((sentence, sentence_score))

    for i in range(0, len(sentences_scores)):
        for j in range(i + 1, len(sentences_scores)):
            if sentences_scores[i][1] < sentences_scores[j][1]:
                aux = sentences_scores[i]
                sentences_scores[i] = sentences_scores[j]
                sentences_scores[j] = aux

    predicted_sentences = []
    k = avg_summary_sent_perc[category] * len(sentences_scores)
    k = int(k)
    for i in range(0, k):
        predicted_sentences.append(sentences_scores[i][0])

    predicted_summary = ""
    for predicted_sentence in predicted_sentences:
        predicted_summary = predicted_summary + " " + predicted_sentence
    
    return predicted_summary

def TF_IDF_Fourgrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html):
    files = GetPaths(shuffle)
    (trainData, testData) = GetDataSets(files)
    idf_vocabulary = GetFourgramsIDF(trainData, stop, lemma, nouns)
    avg_summary_sent_perc = GetAvgSummarySentPerc(trainData)
    trainingResults = (idf_vocabulary, stop, lemma, nouns, title_const, doc_location, avg_summary_sent_perc)
    for category in categories:
        if format_html:
            print("<tr>")
            print("<td>" +  "4-grame" + "</td>") # n-grame
            print("<td>" +  category + "</td>") # categorie
            if nouns: #doar substantive
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
            if title_const > 0: # pondere titlu
                print("<td>" +  "DA" + "</td>")
            else: 
                print("<td>" +  "NU" + "</td>")
            print("<td>" +  str(title_const) + "</td>")
            if doc_location: # pondere locatie document
                print("<td>" +  "DA" + "</td>")
            else:
                print("<td>" +  "NU" + "</td>")
        evaluate(testData, trainingResults, category, PredictSummary_TFIDF_Fourgrams, format_html)
        if format_html:
            print("</tr>")


#--------------------------------------------------------------------------------------------------------------


stop = True
lemma = True
nouns = True
doc_location = False
title_const = 0.3
shuffle = False
format_html = True

#NaiveBayes4grams(stop, lemma, shuffle, format_html)
#TF_IDF_Words(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)


#print("=========================Naive Bayes Unigrame============================")
stop = False
lemma = False
NaiveBayesWords(stop, lemma, shuffle, format_html)
stop = True
NaiveBayesWords(stop, lemma, shuffle, format_html)
lemma = True
NaiveBayesWords(stop, lemma, shuffle, format_html)
#print("=========================Naive Bayes Bigrame=============================")
stop = False
lemma = False
NaiveBayesBigrams(stop, lemma, shuffle, format_html)
stop = True
NaiveBayesBigrams(stop, lemma, shuffle, format_html)
lemma = True
NaiveBayesBigrams(stop, lemma, shuffle, format_html)
#print("=========================Naive Bayes 4grame=============================")
stop = False
lemma = False
NaiveBayes4grams(stop, lemma, shuffle, format_html)
stop = True
NaiveBayes4grams(stop, lemma, shuffle, format_html)
lemma = True
NaiveBayes4grams(stop, lemma, shuffle, format_html)

#print("=========================TF-IDF Unigrame================================")
nouns = False
title_const = 0
doc_location = False
TF_IDF_Words(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
nouns = True
TF_IDF_Words(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
title_const = 0.3
TF_IDF_Words(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
doc_location = True
TF_IDF_Words(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
#print("=========================TF-IDF Bigrame================================")
nouns = False
title_const = 0
doc_location = False
TF_IDF_Bigrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
nouns = True
TF_IDF_Bigrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
title_const = 0.3
TF_IDF_Bigrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
doc_location = True
TF_IDF_Bigrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
#print("=========================TF-IDF 4grame================================")
nouns = False
title_const = 0
doc_location = False
TF_IDF_Fourgrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
nouns = True
TF_IDF_Fourgrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
title_const = 0.3
TF_IDF_Fourgrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
doc_location = True
TF_IDF_Fourgrams(stop, lemma, nouns, doc_location, title_const, shuffle, format_html)
