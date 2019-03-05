from utils import *
from vae_lstm import *
import numpy as np
import pickle


# with open('data/nltk_corpuses_vectorized.p', 'rb') as f:
# 	data = pickle.load(f)

data, all_text = get_data()

len_train = 20000
len_test = 5000
train = data[:len_train]
train_text = all_text[:len_train]
test = data[len_train:len_train + len_test]

batch_size = 50
epochs = 5
input_dim = train.shape[-1]
timesteps = train.shape[1]

model = VAE_LSTM(input_dim=input_dim, latent_dim=100, hidden_dims=[32], timesteps=timesteps, batch_size=batch_size)
vae, encoder, generator = model.autoencoder, model.encoder, model.generator

#train
vae.fit(train, train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(test, test))

encoded_sentences = encoder.predict(np.array(train), batch_size = batch_size)
decoded_sentences = generator.predict(encoded_sentences)

import plotly
import plotly.graph_objs as go
from sklearn.manifold import TSNE

plotly.offline.init_notebook_mode(connected=True)

sample_indices = np.random.choice(len(encoded_sentences), 1000)
sample_embeddings = encoded_sentences[sample_indices]
sample_text = list(np.array(train_text)[sample_indices])
tsne = TSNE()
tsne_embeddings = tsne.fit_transform(sample_embeddings)

# Plot the embeddings in 2D
trace = go.Scatter(
    x = tsne_embeddings[:,0],
    y = tsne_embeddings[:,1],
    mode = 'markers',
    text = sample_text
)

data = [trace]

plotly.offline.plot(data, filename='tsne_vae_encodings.html')