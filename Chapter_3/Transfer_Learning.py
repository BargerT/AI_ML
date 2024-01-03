from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from google.cloud import storage

storage_client = storage.Client("AI_ML")
bucket = storage_client.get_bucket("Transfer_Learning")
blob = bucket.blob("")

weights_url = "https://storage.googleapis.com/mledu-datasets/inceptions_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)