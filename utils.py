from nltk.tokenize import sent_tokenize
from nltk.corpus import reuters, gutenberg, brown, treebank
from gensim.models import KeyedVectors
from scipy import spatial
import numpy as np
import itertools

w2v = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec')

def get_data(lstm_format=True):
	data = []
	all_text = []
	maxlen = 20 if lstm_format else 10
	minlen = 15 if lstm_format else 10
	for t in [brown.words(), reuters.words(), gutenberg.words()]:
		text = get_sentences(t)
		vect, sentences_text = vectorize_sentences(text, maxlen, minlen)
		all_text += sentences_text
		if lstm_format:
			if data == []:
				data = vect
			else:
				data = np.vstack([data, vect])
		else:
			for x in vect:
				data.append(list(itertools.chain.from_iterable(x)))
	data = np.array(data)
	p = np.random.permutation(len(data))
	np.random.shuffle(data)
	return data[p], list(np.array(all_text)[p])

def get_sentences(text):
	"""
	Clean the text and then use NLTK to split it into sentences.
	"""
	s = ''
	for word in text:
		s += word
		s += ' '
	s_cleaned = s.lower()
	for x in ['\xd5d','\n', '\t', '"',"!", '#','$','%','&','(',')','^','*','+',',','-','/',':',';','<','=','>','?','@','[','^',']','_','`','{','|','}','~']:
		s_cleaned = s_cleaned.replace(x, '')
	sentences = sent_tokenize(s_cleaned)
	return sentences

def vectorize_sentences(sentences, maxlen=20, minlen=10):
	"""
	Convert sentence array into word embedding representation.

	PARAMS
	------
	sentences: array of strings; the sentences to be converted to word embedding representation.
	maxlen: int; word length of each sentence in returned np.array. Longer sentences
		will be skipped, and shorter sentences will be padded with zero vectors to be this length.
	minlen: int; minimum word length for sentences to be used in the embedding representation.
		Shorter sentences will be skipped.

	RETURNS
	-------
	3d numpy array of shape (N, maxlen, V), where N is the number of sentences and V is the length of the word embedding
	"""
	vectorized = []
	text = []
	for sentence in sentences:
		words = sentence.split()
		concat_vector = []
		if len(words) > maxlen or len(words) < minlen:
			continue
		elif len(words) < maxlen:	# Zero pad the beginning
			for _ in range(maxlen - len(words)):
				concat_vector.append(np.zeros((300,), dtype=np.float32))
		for word in words:
			try:
				concat_vector.append(w2v[word])
			except:
				concat_vector.append(np.zeros((300,), dtype=np.float32))
				pass
		text.append(sentence)
		vectorized.append(np.array(concat_vector))
	return np.array(vectorized), text

def sentence_embedding_to_text(sentence_vec):
	"""
	Convert sentence embedding to a string using Gensim word2vec
	"""
	text = ''
	for wordvec in sentence_vec:
		if np.all(wordvec == 0):
			continue
		text += w2v.most_similar(positive=[wordvec], topn=1)[0][0]
		text += ' '
	return text

def get_most_similar_encodings(sentence_vec, encoded_sentences, topn=1):
	"""
	Get topn most similar encoded sentences.
	"""
	cos_dists = []
	for sent in encoded_sentences:
		result = spatial.distance.cosine(sentence_vec, sent)
		cos_dists.append(result)
	data_array = np.array(cos_dists)
	maximum_indices = data_array.argsort()[:topn+1][1:]
	new_vecs = encoded_sentences[maximum_indices]
	return new_vecs, maximum_indices

def get_nearest_sentences(sent_idx, encoded_sentences, sentence_embeddings, topn=5):
	"""
	Get the text of the most similar sentences to the given sentence index based on the distance between VAE encodings.
	"""
	sentences = []
	sentences.append(sentence_embedding_to_text(sentence_embeddings[sent_idx]))
	_, neighbor_indices = get_most_similar_encodings(encoded_sentences[sent_idx], encoded_sentences, topn=topn)
	for i in neighbor_indices:
		sentences.append(sentence_embedding_to_text(sentence_embeddings[i]))
	return sentences