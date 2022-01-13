import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from sklearn import preprocessing

from CoughVidData import CoughVidDataset
from model import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
LABEL_ENCODER = preprocessing.LabelEncoder()
LABELS = ['COVID-19', 'healthy']

# Change it to where you have stored your csv and audio files.
ANNOTATIONS_FILE = "./Data/Segmented/combined_segmented.csv"
AUDIO_DIR = "D:/HackerStuff/Project_2_AIML/Data/Segmented"  # This one needs to be absolute path
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    """
    Loads the data in batches.
    :param train_data: Data to be loaded
    :param batch_size: The size of the batch
    :return: DataLoader
    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        target = LABEL_ENCODER.fit_transform(target)
        target = torch.as_tensor(target)
        input, target = input.to(device), target.to(device)

        # Calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # Backpropagate error and update the weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-" * 10)

    print("Finished training")


if __name__ == "__main__":

    # Check if the code should be ran on CPU or GPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    coughVidData = CoughVidDataset(ANNOTATIONS_FILE,
                                   AUDIO_DIR,
                                   mel_spectrogram,
                                   SAMPLE_RATE,
                                   NUM_SAMPLES,
                                   device
                                   )
    coughVidData.use_sampling_technique_on_dataframe()

    train_dataloader = create_data_loader(coughVidData, BATCH_SIZE)

    model = CNNNetwork().to(device)
    print(model)

    # Initialize the loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    torch.save(model.state_dict(), "initial_CNN_model.pth")
    print("Trained CNN saved to initial_CNN_model.pth")
