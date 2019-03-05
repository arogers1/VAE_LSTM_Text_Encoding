import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Layer
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives

class VAE:
	def __init__(self, input_dim, latent_dim, hidden_dims, batch_size, optimizer='rmsprop', epsilon_std = .01):
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims
		self.batch_size = batch_size
		self.optimizer = optimizer
		self.epsilon_std = epsilon_std
		self.build_model()

	def build_model(self):
		input_layer = Input(batch_shape=(self.batch_size, self.input_dim))
		self.build_encoder(input_layer)
		self.build_decoder()
		self.autoencoder = Model(input_layer, self.x_decoded_mean)
		vae_loss = self._get_vae_loss()
		self.autoencoder.compile(optimizer=self.optimizer, loss=vae_loss)

	def build_encoder(self, input_layer):
		prev_layer = input_layer
		for q in self.hidden_dims:
			hidden = Dense(q, activation='relu')(prev_layer)
			prev_layer = hidden
		self._build_z_layers(hidden)
		self.encoder = Model(input_layer, self.z_mean)

	def _build_z_layers(self, hidden_layer):
		self.z_mean = Dense(self.latent_dim)(hidden_layer)
		self.z_log_sigma = Dense(self.latent_dim)(hidden_layer)

	def build_decoder(self):
		z = self._get_sampling_layer()
		prev_layer = z
		for q in self.hidden_dims:
			hidden = Dense(q, activation='relu')(prev_layer)
			prev_layer = hidden
		self.x_decoded_mean = Dense(self.input_dim, activation='sigmoid')(prev_layer)

		# Build the stand-alone generator
		generator_input = Input((self.latent_dim,))
		prev_layer = generator_input
		for q in self.hidden_dims:
			hidden = Dense(q, activation='relu')(prev_layer)
			prev_layer = hidden
		gen_x_decoded_mean = Dense(self.input_dim, activation='sigmoid')(prev_layer)
		self.generator = Model(generator_input, gen_x_decoded_mean)

	def _get_sampling_layer(self):
		def sampling(args):
			z_mean, z_log_sigma = args
			epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
									  mean=0., stddev=self.epsilon_std)
			return z_mean + z_log_sigma * epsilon
		return Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])

	def _get_vae_loss(self):
		z_log_sigma = self.z_log_sigma
		z_mean = self.z_mean
		def vae_loss(x, x_decoded_mean):
			reconstruction_loss = objectives.mse(x, x_decoded_mean)
			kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
			return reconstruction_loss + kl_loss

		return vae_loss


class VAE_LSTM(VAE):
	def __init__(self, input_dim, latent_dim, hidden_dims, timesteps, batch_size, optimizer='rmsprop', epsilon_std = .01):
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims
		self.batch_size = batch_size
		self.timesteps = timesteps
		self.optimizer = optimizer
		self.epsilon_std = epsilon_std
		self.build_model()

	def build_model(self):
		input_layer = Input(shape=(self.timesteps, self.input_dim,))
		self.build_encoder(input_layer)
		self.build_decoder()
		self.autoencoder = Model(input_layer, self.x_decoded_mean)
		vae_loss = self._get_vae_loss()
		self.autoencoder.compile(optimizer=self.optimizer, loss=vae_loss)

	def build_encoder(self, input_layer):
		prev_layer = input_layer
		for q in self.hidden_dims:
			hidden = LSTM(q)(prev_layer)
			prev_layer = hidden
		self._build_z_layers(hidden)
		self.encoder = Model(input_layer, self.z_mean)

	def build_decoder(self):
		z = self._get_sampling_layer()
		prev_layer = RepeatVector(self.timesteps)(z)
		for q in self.hidden_dims:
			hidden = LSTM(q, return_sequences=True)(prev_layer)
			prev_layer = hidden
		self.x_decoded_mean = LSTM(self.input_dim, return_sequences=True)(prev_layer)

		# Build the stand-alone generator
		generator_input = Input((self.latent_dim,))
		prev_layer = RepeatVector(self.timesteps)(generator_input)
		for q in self.hidden_dims:
			hidden = LSTM(q, return_sequences=True)(prev_layer)
			prev_layer = hidden
		gen_x_decoded_mean = LSTM(self.input_dim, return_sequences=True)(prev_layer)
		self.generator = Model(generator_input, gen_x_decoded_mean)