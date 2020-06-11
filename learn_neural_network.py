import csv
import re
import gensim
import pymorphy2
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle


def load_file(file_name, observer):
	"""Загрузка файла с данными

	Args:
		file_name (string): имя файлы
		observer (func): обработчик для строки
	"""
	file_obj = open(file_name, "r")
	for line in file_obj:
		observer(line)

def clear_null_bytes(file_name):
	"""Очистка строки от NULL-bytes символов

		Args:
			file_name (string): наименование файла
	"""
	data = open(file_name, 'r').read()
	f = open('clear.data', 'w')
	data = data.replace('\x00', '')
	f.write(data)
	f.close()

def clear_twitter_message(message):
	"""Очистка текста твита. Оставляем только русские слова и выражения

	Args:
		message (string): твит сообщения

	Returns:
		message (string): очищенный твит
	"""
	global LIST
	message = message.strip().lower()
	LIST.append(gensim.utils.simple_preprocess(message))

def build_word2vec_model(documents):
	"""Документы. Список предложений

		Args:
			documents (list<string>): список документов

		Returns:
			word2vec (object): модель
	"""
	model = gensim.models.Word2Vec(documents,size=150, window=10, min_count=2, workers=10, iter=100)
	return model

def to_list_messages(message):
	global LIST
	message = re.sub(r'\n', ' ', message[3])
	message = re.sub(r'RT', '', message)
	message = re.sub(r'htt[^\s]+', '', message)
	message = re.sub(r'#[^\s]+', '', message)
	message = re.sub(r'(^|\s)@($|\s)', '', message)
	message = re.sub(r'\@[^\s]+', 'Name', message)
	message = re.sub(r'\s+', ' ', message)
	message = message.strip().lower()
	LIST.append(message)

def lemma_file(source_file, dest_file):
	"""
	Лемматизация сообщений с сохранением в файл

	Args:
		source_file (string): наименование файла источника
		dest_file (string): наименование файла назначения
	                    
	"""
	file = open(source_file, "r")
	save_file = open(dest_file, "w")
	morph = pymorphy2.MorphAnalyzer()
	for message in file.read().split("\n"):
		lemma = lemma_message(message, morph)
		save_file.write(lemma + "\n")
		print(lemma)
	file.close()
	save_file.close()
	print("Lemmatization end")

def lemma_message(message, morph = None):
	"""
	Лемматизировать сообщение

	Args:
		message (string): сообщение

	Returns:
		(string): лемматизированное сообщение
	"""
	if not morph:
		morph = pymorphy2.MorphAnalyzer()
	lemma = []
	message = re.sub(r"[\(\)\{\}\[\],\.:;\+\-]", "", message)
	for word in message.split(" "):
		res = morph.parse(word)[0]
		if res.normal_form:
			res = res.normal_form
		else:
			res = word 
		lemma.append(res)
	return " ".join(lemma)

def get_dataset(positive_file_name, negative_file_name):
	"""
	Загрузить датасет позитивных и негативных сообщений

	Args:
		positive_file_name(string): наименование файла с позитивными сообщениями
		negative_file_name(string): наименование файла с негативными сообщениями

	Returns:
		(list, list): датасет данных + вектор тональностей
	"""
	positive_list = open("positive_list.data").read().split("\n")
	negative_list = open("negative_list.data").read().split("\n")
	full_list = positive_list + negative_list
	marks = [1 for p in positive_list] + [0 for n in negative_list]
	return (full_list, marks)

def get_vectorizer():
	"""
	Получить векторизатор

	Args:
		dataset (list<string>): список сообщений

	Returns:
		vectorizer (object): объект векторизатор 
	"""
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
	return vectorizer

def get_classifier(train_data_features, marks):
	"""
	Обучить и получить классификатор

	Args:
		train_data_features (list<string>): датасет сообщений

	Returns:
		 
	"""
	forest = RandomForestClassifier(n_estimators = 100) 
	forest = forest.fit(train_data_features, marks)
	return forest

def get_dataset_and_learn_neural_network():
	"""
	Загрузить датасет и обучить нейронную сеть 
	
	"""
	positive_list = open("positive_list.data").read().split("\n")
	negative_list = open("negative_list.data").read().split("\n")

	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 

	dataset = positive_list + negative_list
	marks = [1 for p in positive_list] + [0 for n in negative_list]
	train_data_features = vectorizer.fit_transform(dataset)
	train_data_features = train_data_features.toarray()

	forest = RandomForestClassifier(n_estimators = 100) 
	forest = forest.fit(train_data_features, marks)

	forest_dump = open("forest_dump.object", "wb")
	pickle.dump(forest, forest_dump, pickle.HIGHEST_PROTOCOL)
	forest_dump.close()
	vect_dump = open("vect_dump.object", "wb")
	pickle.dump(vectorizer, vect_dump, pickle.HIGHEST_PROTOCOL)
	vect_dump.close()

if __name__ == '__main__':
	#load_file("negative_list.data", clear_twitter_message)
	#build_word2vec_model(LIST)
	#lemma_file("positive_list.data", "lemma_positive_list.data")
	#lemma_file("negative_list.data", "lemma_negative_list.data")
	if False:
		dataset = get_dataset("lemma_positive_list.data", "lemma_negative_list.data")
		vectorizer = get_vectorizer(dataset)
		train_data_features = vectorizer.fit_transform(dataset[0])
		train_data_features = train_data_features.toarray()
		classifier = get_classifier(train_data_features, dataset[1])
		classifier_dump = open("classifier_dump.object", "wb")
		pickle.dump(classifier, classifier_dump, pickle.HIGHEST_PROTOCOL)
		classifier_dump.close()
		vect_dump = open("vect_dump.object", "wb")
		pickle.dump(vectorizer, vect_dump, pickle.HIGHEST_PROTOCOL)
		vect_dump.close()
	file_classifier = open("classifier_dump.object", "rb")
	classifier = pickle.load(file_classifier)
	file_vect = open("vect_dump.object", "rb")
	vect = pickle.load(file_vect)
	file_classifier.close()
	file_vect.close()
	d = vect.transform([lemma_message("ужасный день")])[0]
	print( classifier.predict(d) )



