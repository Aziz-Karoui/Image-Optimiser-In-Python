import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# model architecture
class ImageEnhancementModel(nn.Module):
    def __init__(self):
        super(ImageEnhancementModel, self).__init__()

        # Define the layers here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        # forward pass
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
        
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image)
        return image


# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 50

# Initialize your model
model = ImageEnhancementModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader for training data
train_dataset = CustomDataset(data_dir='before')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        # Forward pass
        outputs = model(data)
        
        # Load the corresponding "after enhancement" images
        target_data = CustomDataset(data_dir='after')  # Load the "after" images
        target_data = next(iter(target_data))  # Get the corresponding target image

        loss = criterion(outputs, target_data)  # Use the "after" images as targets

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'image_enhancement_model.pth')

# Inference (enhance images)
model.eval()  # Set the model to evaluation mode

# Load and preprocess an input image
input_image = Image.open('testb.jpg')
input_image = train_dataset.transform(input_image).unsqueeze(0)

# Use the trained model to enhance the input image
enhanced_image = model(input_image)

# Save 
output_image = enhanced_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_image = (output_image + 1) / 2.0 * 255.0  # Denormalize
output_image = output_image.astype('uint8')
Image.fromarray(output_image).save('enhanced_image.jpg')
