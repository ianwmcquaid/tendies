import argparse
import sys
import tensorflow as tf
import numpy as np
from client import Client

# sys.path.insert(0, "C:\\Users\\Tyler Labonte\\Desktop\\models\\research\\object_detection\\utils")  # nopep8
# import visualization_utils  # nopep8
# import label_map_util  # nopep8


class ObjectDetectionClient(Client):
    """ Object Detection API compliant client for a TensorFlow ModelServer.

        Performs inference on a directory of images by sending them
        to a TensorFlow-Serving ModelServer, using its RESTful API.
    """

    def __init__(self,
                 url,
                 input_dir,
                 input_extension,
                 output_dir,
                 output_filename,
                 encoding,
                 channels,
                 label_path):
        """ Initializes an ObjectDetectionClient object.

            Args:
                url: The URL of the TensorFlow ModelServer.
                input_dir: The name of the input directory.
                input_extension: The file extension of input files.
                output_dir: The name of the output directory.
                output_filename: The filename (less extension) of output files.
                output_extension: The file extension of output files.
                encoding: The type of string encoding to be used.
                image_size: The size of the input images.
                label_path: The path to the label mapping file.
        """

        # Initializes a Client object
        super().__init__(url,
                         input_dir,
                         input_extension,
                         output_dir,
                         output_filename,
                         encoding)
        # Adds child class specific member variables
        self.channels = channels
        self.label_path = label_path

    def get_category_index(self):
        """ Transforms label map into category index for visualization.

            Returns:
                category_index: The category index corresponding to the given
                    label map.
        """
        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(
                        label_map,
                        max_num_classes=1,
                        use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def visualize(self, input_image, response, i):
        """ Decodes JSON data and converts it to a bounding box overlay
            on the input image, then saves the image to a directory.

            Args:
                input_image: The string representing the input image.
                response: The list of response dictionaries from the server.
                i: An integer used in iteration over input images.
        """

        # Processes response for visualization
        detection_boxes = response["detection_boxes"]
        detection_classes = response["detection_classes"]
        detection_scores = response["detection_scores"]

        # Converts image bitstring to uint8 tensor
        input_bytes = tf.reshape(input_image, [])
        image = tf.image.decode_jpeg(input_bytes, channels=self.channels)

        # Gets value of image tensor
        with tf.Session() as sess:
            image = image.eval()

        # Overlays bounding boxes and labels on image
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            np.asarray(detection_boxes, dtype=np.float32),
            np.asarray(detection_classes, dtype=np.uint8),
            scores=np.asarray(detection_scores, dtype=np.float32),
            category_index=self.get_category_index(),
            use_normalized_coordinates=True,
            line_thickness=2)

        # Saves image
        output_file = self.output_dir + "/images/"
        output_file += self.output_filename + str(i) + ".png"
        visualization_utils.save_image_array_as_png(image, output_file)


def example_usage(_):
    # Instantiates an ObjectDetectionClient
    object_detection_client = ObjectDetectionClient(FLAGS.url,
                                                    FLAGS.input_dir,
                                                    FLAGS.input_extension,
                                                    FLAGS.output_dir,
                                                    FLAGS.output_filename,
                                                    FLAGS.encoding,
                                                    FLAGS.channels,
                                                    FLAGS.label_path)
    # Performs inference
    object_detection_client.inference()


if __name__ == "__main__":
    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--url",
                        type=str,
                        default="http://localhost:8501/v1/models/"
                                "saved_model:predict",
                        help="URL of server")

    parser.add_argument("--input_dir",
                        type=str,
                        default="input",
                        help="Path to input directory")

    parser.add_argument("--input_extension",
                        type=str,
                        default=".jpg",
                        help="Input file extension")

    parser.add_argument("--output_dir",
                        type=str,
                        default="output",
                        help="Path to output directory")

    parser.add_argument("--output_filename",
                        type=str,
                        default="output",
                        help="Output file name")

    parser.add_argument("--encoding",
                        type=str,
                        default="utf-8",
                        help="Encoding type")

    parser.add_argument("--channels",
                        type=int,
                        default=3,
                        help="Image channels")

    parser.add_argument("--label_path",
                        type=str,
                        default="C:\\Users\\Tyler Labonte\\Desktop\\rcnn\\label_data\\astronet_label_map_2.pbtxt",
                        help="Label map path")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the inference
    example_usage([sys.argv[0]] + unparsed)
