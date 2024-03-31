from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray

image = Image.open('data/MNIST/raw/test/0.jpg')  # change this according to your
# dataset directory
img_np = asarray(image)
img_np_flat = img_np.flatten()
text = img_np_flat
print("pixels: ", img_np_flat.tolist())
print("len(img_np_flat): ", len(img_np_flat))

tokenizer = BasicTokenizer()

# tokenizer.train(text, 256 + 50, 2, verbose=True)  # 256 are the byte tokens, then do 50 merges
# tokenizer.save("toy")  # writes two files: toy.model (for loading) and toy.vocab (for viewing)
tokenizer.load('models/basic.model')  # load trained model

output_encoder = tokenizer.encode(text)
print("output_encoder: ", output_encoder)
print("len(output_encoder): ", len(output_encoder))

output_decoder = tokenizer.decode(tokenizer.encode(text))
print("output_decoder: ", output_decoder)
print("len(output_decoder): ", len(output_decoder))
print("pixels == output_decoder ? ", img_np_flat.tolist() == output_decoder)


