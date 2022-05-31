import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import random
from pathlib import Path


class Model(nn.Module):
    """
    A U-Net style convolutional neural network for image denoises
    Implement a Noise2Noise model
    """
    def __init__(self, channels=3):
        super().__init__()
        """
        Parameters:
            channels (int): input tensor channel size (default: 3)
        """

        self.encode1 = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        self.encode2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))

        self.decode1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1))
        self.decode2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
        self.decode3 = nn.Sequential(
            nn.Conv2d(32 + channels, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1))

        self.output_layer = nn.Conv2d(16, channels, 3, stride=1, padding=1)

        self.loss_function = nn.MSELoss()

    def forward(self, x):
        """
        forward pass
        Args:
            x: input data
        Returns:
            output of the network
        """
        # layers
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.decode1(pool2)

        concat3 = torch.cat((pool3, pool1), dim=1)
        upsample2 = self.decode2(concat3)
        concat2 = torch.cat((upsample2, x), dim=1)
        upsample1 = self.decode3(concat2)
        output = self.output_layer(upsample1)

        return output

    def train(self, train_input, train_target, num_epochs):
        """
        train the model
        Args:
            train_input: tensor of size (N, C, H, W) containing a noisy version of the images
            train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, 
                          which only differs from the input by their noise
            num_epochs (int): number of epoches to train the model
        """
        # device: cpu or gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.to(device)
        print('device:',device)
        
        # parameters
        epoch_train_loss = []
        
        # bacth size is 16 for fast testing, but 4 is used when training
        BATCH_SIZE = 16
        LEARNING_RATE = 0.0004
        
        # optimizer
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        
        # normalization
        train_input_normalized = train_input / 255
        train_target_normalized = train_target / 255
        
        # batch
        train = TensorDataset(train_input_normalized, train_target_normalized)
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
        
        for epoch in range(num_epochs):
            training_loss = 0
            
            for data, target in train_loader:
                if torch.cuda.is_available():
                    data = data.to(device)
                    target = target.to(device)

                # data, target = self.data_augmentation(data, target)
                optimizer.zero_grad()

                # forward + backward + optimize
                output = net(data)
                loss = net.loss_function(output, target)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

            epoch_train_loss.append(training_loss/train_input_normalized.size(0))

        return epoch_train_loss

    def predict(self, test_input):
        """
        Use the trained network to predict the output of test input
        Args:
            test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        Returns:
            output: the predicted clean images for input.
        """
        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.to(device)
        
        # prediction
        test_input_normalized = test_input / 255
        output = net(test_input_normalized.to(device))
        output = torch.clamp(output, 0, 1)
        output = output * 255
        output = output.to("cpu")

        return output

    def load_pretrained_model(self):
        """
        Load the pretrained best model
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.load_state_dict(torch.load(model_path))

    def data_augmentation(self, train, target):
        """
        Data augmentation: increasing dataset by adding rotation images, grey images and recolored images
        Args:
            train: train data
            target: target data
        Return: 
            augmented train data and target data
        """
        transform1 = transforms.RandomRotation(degrees=180)
        transform2 = transforms.Grayscale(3)
        transform3 = transforms.ColorJitter(brightness=1, contrast=0.5, saturation=0.9, hue=0.3)
        s = random.randint(0, 12345678)
        
        # augmentation on train input
        random.seed(s)
        torch.manual_seed(s)
        aug_train = torch.cat((transform1(train), train), dim=0)
        aug_train = torch.cat((transform2(train), aug_train), dim=0)
        aug_train = torch.cat((transform3(train), aug_train), dim=0)
        
        # augmentation on train target
        random.seed(s)
        torch.manual_seed(s)
        aug_target = torch.cat((transform1(target), target), dim=0)
        aug_target = torch.cat((transform2(target), aug_target), dim=0)
        aug_target = torch.cat((transform3(target), aug_target), dim=0)

        return aug_train, aug_target