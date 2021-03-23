import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
import imageio								# Replacement for deprecated scipy imsave
from skimage.transform import resize		# Replacement for deprecated scipy resize
from scipy.optimize import fmin_l_bfgs_b	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import warnings


'''
------------------------------------------------------------------------------------------------------------------------------------------
Author:
------------------------------------------------------------------------------------------------------------------------------------------
Anurag Shah

------------------------------------------------------------------------------------------------------------------------------------------
External Resources Used:
------------------------------------------------------------------------------------------------------------------------------------------
https://keras.io/api/utils/backend_utils/
	Used for general backend functions

https://keras.io/examples/generative/neural_style_transfer/
	Used for deprocessing and total variation loss as a regularizer

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
	Used mostly to remove the fprime argument, which was causing crashes since gradients were included by kFunction

------------------------------------------------------------------------------------------------------------------------------------------
Links to images:
------------------------------------------------------------------------------------------------------------------------------------------
https://en.wikipedia.org/wiki/Arc_de_Triomphe#/media/File:Arc_de_Triomphe,_Paris_5_February_2019.jpg
https://en.wikipedia.org/wiki/File:Tsunami_by_hokusai_19th_century.jpg
https://en.wikipedia.org/wiki/Colosseum#/media/File:Colosseo_2020.jpg
https://www.outlookindia.com/outlooktraveller/public/uploads/articles/explore/shutterstock_1430024867.jpg
https://uploads5.wikiart.org/images/paul-cezanne/road-near-mont-sainte-victoire.jpg!Large.jpg
https://en.wikipedia.org/wiki/The_Starry_Night#/media/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg

------------------------------------------------------------------------------------------------------------------------------------------
Notes:
------------------------------------------------------------------------------------------------------------------------------------------
Hyperparameters were tuned individually for the images in question, the ones in the file below are not meant to be global for the project

------------------------------------------------------------------------------------------------------------------------------------------
'''

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONTENT_IMG_PATH = "input_images/colosseum.png"
STYLE_IMG_PATH = "input_images/starrynight.png"


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 4e-5					# Alpha weight.
STYLE_WEIGHT = 1 - CONTENT_WEIGHT		# Beta weight.
TOTAL_WEIGHT = 2e-6						# Regularization weight.

TRANSFER_ROUNDS = 20

# Allow the use of gradient()
tf.compat.v1.disable_eager_execution()

#=============================<Helper Fuctions>=================================

def deprocessImage(img):
	final = np.copy(img)

	# Change the shape to image (500, 500, 3) instead of like (1, 500, 500 3)
	final = final.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))

	# VGG19 preprocesses the image to be zero centered wrt imagenet data, and in BGR
	# So we reverse this, and then clip it back to integer values
	final[:, :, 0] += 103.939
	final[:, :, 1] += 116.779
	final[:, :, 2] += 123.68
	final = final[:, :, ::-1]
	final = np.clip(final, 0, 255).astype("uint8")
	return final


def gramMatrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
	return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4.0 * ((STYLE_IMG_W * STYLE_IMG_H * 3) ** 2))


def contentLoss(content, gen):
	return K.sum(K.square(gen - content))


def totalLoss(x):
	# Total Variation Loss: acts as a regularizer
	a = K.square(x[:, : STYLE_IMG_H - 1, : STYLE_IMG_W - 1, :] - x[:, 1:, : STYLE_IMG_W - 1, :])
	b = K.square(x[:, : STYLE_IMG_H - 1, : STYLE_IMG_W - 1, :] - x[:, : STYLE_IMG_H - 1, 1:, :])
	return tf.reduce_sum(K.pow(a + b, 1.25))


#=========================<Pipeline Functions>==================================

def getRawData():
	print("   Loading images.")
	print("	  Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
	print("	  Style image URL:	\"%s\"." % STYLE_IMG_PATH)
	cImg = load_img(CONTENT_IMG_PATH)
	tImg = cImg.copy()
	sImg = load_img(STYLE_IMG_PATH)
	print("	  Images have been loaded.")
	return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
	img, ih, iw = raw
	img = img_to_array(img)
	# Editied to use skimage instead
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		img = resize(img, (ih, iw, 3), anti_aliasing=True)
	img = img.astype("float64")
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)
	return img


def styleTransfer(cData, sData, tData):
	print("   Building transfer model.")
	contentTensor = K.variable(cData, dtype=tf.float64)
	styleTensor = K.variable(sData, dtype=tf.float64)
	# For some reason if I create genTensor first and flatten it gradients breaks, but this way works
	# Can't directly use genTensor because of conflicting shapes with grad, which you cant flatten
	# Just doing it this way instead of passing functions to the optimizer makes it a ton more efficient though
	# Also same reason why tData needs to be flattened later on
	genTensor_flattened = K.placeholder(CONTENT_IMG_H * CONTENT_IMG_W * 3, dtype=tf.float64)
	genTensor = K.reshape(genTensor_flattened, (1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
	inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)

	model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)

	outputDict = dict([(layer.name, layer.output) for layer in model.layers])

	print("   VGG19 model loaded.")

	loss = 0.0
	styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
	contentLayerName = "block5_conv2"

	print("   Calculating content loss.")

	contentLayer = outputDict[contentLayerName]
	contentOutput = contentLayer[0, :, :, :]
	genOutput = contentLayer[2, :, :, :]

	# Calculate content loss
	loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)

	print("   Calculating style loss.")

	# Calculate style loss for every style layer
	for layerName in styleLayerNames:
		curr_layer = outputDict[layerName]
		loss += STYLE_WEIGHT * styleLoss(curr_layer[1, :, :, :], curr_layer[2, :, :, :])
	
	# Calculate total variation loss
	loss += TOTAL_WEIGHT * totalLoss(tf.cast(genTensor, tf.float32))
	
	# Compute the gradients off the flattened tensor
	grads = K.gradients(loss, genTensor_flattened)
	outputs = [loss]
	outputs += grads

	# Create the kFunction, for loss function for gradient descent
	kFunction = K.function([genTensor_flattened], outputs)
	genimg = tData.flatten()

	print("   Beginning transfer.")
	for i in range(TRANSFER_ROUNDS):
		print("   Step %d." % i)

		# Get the return values from fmin_l_bfgs_b
		returns = fmin_l_bfgs_b(func=kFunction, x0=genimg, maxiter=50)

		# Create a deep copy of the generated image
		genimg = np.copy(returns[0])
		print("	  Loss: %f." % returns[1])

		# Deprocess and save the image
		img = deprocessImage(genimg)
		saveFile = "output_iteration_" +str(i)+".png"
		imageio.imwrite(saveFile, img)   #Uncomment when everything is working right.
		print("	  Image saved to \"%s\"." % saveFile)
	print("   Transfer complete.")


#=========================<Main>================================================

def main():
	print("Starting style transfer program.")
	raw = getRawData()
	cData = preprocessData(raw[0])   # Content image.
	sData = preprocessData(raw[1])   # Style image.
	tData = preprocessData(raw[2])   # Transfer image.
	styleTransfer(cData, sData, tData)
	print("Done. Goodbye.")



if __name__ == "__main__":
	main()
