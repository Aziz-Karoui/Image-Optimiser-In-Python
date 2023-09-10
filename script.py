from PIL import Image, ImageEnhance

# Open the input image
image = Image.open('dog.jpg')

# Resize the image
width, height = image.size
target_width = width * 2
target_height = height * 2
resized_image = image.resize((target_width, target_height), Image.BILINEAR)

# Apply image enhancements
enhancer = ImageEnhance.Brightness(resized_image)
enhanced_image = enhancer.enhance(1)  # Adjust brightness (increase or decrease the value as desired)

# Save the final enhanced image
enhanced_image.save('dogii.jpg')
