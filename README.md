# Neural Style Transfer (NST) for Image Enhancement

Enhance your images using Neural Style Transfer by combining the content of an input image with the style of a reference image.

## Description

This project uses TensorFlow to perform Neural Style Transfer (NST) on an input image using a style reference image. NST is a technique for enhancing an image by transferring the artistic style of one image (the reference style image) to the content of another image (the input image). The result is a new image that combines the content of the input image with the artistic style of the reference image.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow

You can install the required Python packages by running:


## Usage

1. Prepare your input image and style reference image and save them in the project directory.

2. Update the paths to your input and style reference images in the script (`input_image_path` and `style_image_path` variables).

3. Run the script:


4. The script will optimize the generated image to combine the content of the input image with the style of the reference image.

5. The final enhanced image will be saved as `enhanced_image.jpg` in the project directory.

## Examples

Here are some example results of using NST to enhance images:

![Input Image](examples/input_image.jpg)
![Style Reference Image](examples/style_image.jpg)
![Enhanced Image](examples/enhanced_image.jpg)

## License

This project is licensed under the Aziz Karoui License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is based on the Neural Style Transfer technique developed by Gatys et al.
- Pre-trained VGG models provided by the Keras team.

Feel free to modify this README file to include more details, usage instructions, or additional sections relevant to your project.
