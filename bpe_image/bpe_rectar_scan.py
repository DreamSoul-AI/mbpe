from tensorflow.keras.datasets import mnist
from collections import Counter, defaultdict


class ConvertMNISTData:
    def __init__(self, image_index=0):
        # Load MNIST data upon class instantiation
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.image_index = image_index  # Index of the image to convert

    def get_image_array(self):
        # Retrieve the specified image from the training dataset
        return self.x_train[self.image_index]

    def convert_image_to_strings(self):
        # Convert the image into a list of lists with each pixel as a string
        image_array = self.get_image_array()
        return [[str(pixel) for pixel in row] for row in image_array]

    def flatten_image(self, image):
        # Flatten the image into a sequence for BPE
        return [str(pixel) for row in image for pixel in row]


class BytePairEncoding:
    def __init__(self, data):
        self.data = data
        self.original_data = list(data)
        self.vocab = Counter(data)
        self.merge_map = {}
        self.merge_count = 1
        self.merge_definitions = {}

    def get_stats(self):
        pairs = Counter()
        for i in range(len(self.data) - 1):
            pair = (self.data[i], self.data[i + 1])
            pairs[pair] += 1
        return pairs

    def get_merge_name(self, pair):
        # Name the merges consistently with a prefix and an incrementing index
        merge_name = f"merge_{self.merge_count}"
        self.merge_definitions[merge_name] = pair
        self.merge_count += 1
        return merge_name

    def merge_vocab(self, pair):
        merge_name = self.get_merge_name(pair)
        new_data = []
        i = 0
        while i < len(self.data):
            if (
                i < len(self.data) - 1
                and self.data[i] == pair[0]
                and self.data[i + 1] == pair[1]
            ):
                new_data.append(merge_name)
                i += 2
            else:
                new_data.append(self.data[i])
                i += 1
        self.data = new_data
        self.vocab = Counter(self.data)

    def apply_bpe(self, num_merges=100):
        for _ in range(num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair)

    def print_named_vocab(self):
        print("Vocabulary:")
        for key, value in self.vocab.items():
            print(f"{key}: {value}")

        print("\nMerge Definitions:")
        for merge_name, components in self.merge_definitions.items():
            print(f"{merge_name}: {components}")

    def decode(self):
        # Initialize the decoded data with the final encoded form
        decoded_data = self.data
        # Reverse the merge definitions to facilitate decoding
        for i in reversed(range(1, self.merge_count)):
            merge_name = f"merge_{i}"
            # Replace each occurrence of the merge token with its original components
            decoded_data = [
                item
                for data_item in decoded_data
                for item in (
                    self.merge_definitions[merge_name]
                    if data_item == merge_name
                    else (data_item,)
                )
            ]
        return decoded_data


def main():
    converter = ConvertMNISTData(image_index=0)
    image_as_strings = converter.convert_image_to_strings()
    flattened_image = converter.flatten_image(image_as_strings)

    bpe = BytePairEncoding(flattened_image)
    bpe.apply_bpe(num_merges=10)
    bpe.print_named_vocab()

    decoded_image = bpe.decode()
    assert (
        converter.flatten_image(image_as_strings) == decoded_image
    ), "Decoded image does not match the original image."
    print("\nDecoding successful. The original image has been reconstructed.")


if __name__ == "__main__":
    main()
