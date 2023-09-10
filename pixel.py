import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Load a pre-trained VGG19 model without the fully connected layers (used for feature extraction)
base_model = VGG19(weights="imagenet", include_top=False)

# Specify the layers to use for style and content representations
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
content_layer = "block4_conv2"

# Create a model that extracts style and content features
style_extractor = Model(inputs=base_model.input, outputs=[base_model.get_layer(layer).output for layer in style_layers])
content_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer(content_layer).output)

# Define a function to compute the Gram matrix for style representation
def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

# Define a custom loss function that computes the style loss
def style_loss(style_targets, predicted_styles):
    loss = 0
    for style_target, predicted_style in zip(style_targets, predicted_styles):
        loss += tf.reduce_mean(tf.square(gram_matrix(style_target) - gram_matrix(predicted_style)))
    return loss

# Define a custom loss function that computes the content loss
def content_loss(content_target, predicted_content):
    return tf.reduce_mean(tf.square(content_target - predicted_content))

# Load your input and style images
input_image_path = "input_image.jpg"
style_image_path = "style_image.jpg"

input_image = tf_image.load_img(input_image_path)
style_image = tf_image.load_img(style_image_path)

input_image = tf_image.img_to_array(input_image)
style_image = tf_image.img_to_array(style_image)

# Preprocess the images (VGG19 requires specific preprocessing)
input_image = tf_image.smart_resize(input_image, (256, 256))
style_image = tf_image.smart_resize(style_image, (256, 256))

input_image = tf_image.img_to_array(input_image)
style_image = tf_image.img_to_array(style_image)

input_image = tf.keras.applications.vgg19.preprocess_input(input_image)
style_image = tf.keras.applications.vgg19.preprocess_input(style_image)

input_image = np.expand_dims(input_image, axis=0)
style_image = np.expand_dims(style_image, axis=0)

# Define a variable to store the generated image and create a TensorFlow variable for it
generated_image = tf.Variable(input_image, dtype=tf.float32)

# Define optimizer and hyperparameters
optimizer = Adam(learning_rate=10.0)

# Number of iterations for optimization
num_iterations = 1000

# Extract style and content features from the style and input images
style_features = style_extractor(style_image)
content_features = content_extractor(input_image)

# Define target style features (the same style for all layers)
style_targets = [style_extractor(tf.constant(style_image)) for _ in style_layers]

# Main optimization loop
for iteration in range(num_iterations):
    with tf.GradientTape() as tape:
        # Extract features from the generated image
        generated_features = style_extractor(generated_image)

        # Compute style loss and content loss
        current_style_loss = style_loss(style_targets, generated_features)
        current_content_loss = content_loss(content_features, generated_features[-1])

        # Total loss as a combination of style and content loss
        total_loss = current_style_loss + current_content_loss

    # Compute gradients
    gradients = tape.gradient(total_loss, generated_image)

    # Update the generated image using the gradients
    optimizer.apply_gradients([(gradients, generated_image)])

    # Clip pixel values to the [0, 255] range
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=255.0))

    # Print the progress
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Total loss: {total_loss}")

# Convert the final generated image to a NumPy array
final_image = tf_image.img_to_array(generated_image[0])

# Clip pixel values to the [0, 255] range and cast to uint8
final_image = np.clip(final_image, 0, 255).astype(np.uint8)

# Save the final image
final_image_path = "enhanced_image.jpg"
tf.keras.preprocessing.image.save_img(final_image_path, final_image[0])

# Display the final enhanced image
plt.imshow(final_image[0])
plt.axis("off")
plt.show()
