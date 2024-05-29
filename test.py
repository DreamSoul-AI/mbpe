from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray

image = Image.open('./0.jpg')
image1 = Image.open('./1.jpg')
img_np = asarray(image)
img_np_flat = img_np.flatten()
img_np1 = asarray(image1)
img_np_flat1 = img_np1.flatten()
print("pixels: ", img_np_flat.tolist())
print("len(img_np_flat): ", len(img_np_flat))
tokenizer = BasicTokenizer()

text = img_np_flat
text1 = img_np_flat1
tokenizer.train(text, 256 + 48,  verbose=True)  # 256 are the byte tokens, then do 48 merges
output_encoder = tokenizer.encode(text)
print("output_encoder: ", output_encoder)
print("len(output_encoder): ", len(output_encoder))

output_decoder = tokenizer.decode(tokenizer.encode(text))
print("output_decoder: ", output_decoder)
print("len(output_decoder): ", len(output_decoder))
print("pixels == output_decoder ? ", img_np_flat.tolist() == output_decoder)

tokenizer.save("toy")
# writes two files: toy.model (for loadinrg) and toy.vocab (for viewing)