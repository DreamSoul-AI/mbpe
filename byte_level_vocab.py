from PIL import Image


def create_byte_level_vocab(img_path):
    # Read the image file as a binary stream
    with open(img_path, "rb") as image_file:
        # Read all bytes from the file
        image_bytes = image_file.read()

    print(image_bytes)
    # Create a set to store unique bytes
    byte_vocab = set(image_bytes)

    # Convert the set to a list for indexing
    byte_vocab_list = list(byte_vocab)

    return byte_vocab_list


# Example usage
image_path = "0.jpg"
byte_vocabulary = create_byte_level_vocab(image_path)

# Print the byte-level vocabulary
# print(byte_vocabulary)
