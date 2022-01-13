import numpy as np
import torch
import torchaudio
from sklearn.metrics import f1_score

from CoughVidData import CoughVidDataset
from trainingModel import CNNNetwork, SAMPLE_RATE, ANNOTATIONS_FILE, AUDIO_DIR, NUM_SAMPLES

class_mapping = [
    "COVID-19",
    "healthy"
]


def predict(model, input, target, class_mapping):
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = target

    return predicted, expected


if __name__ == "__main__":
    cnn = CNNNetwork()

    # Loads the model's weights and parameters.
    state_dict = torch.load("initial_CNN_model.pth")
    cnn.load_state_dict(state_dict)

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
                                   "cpu"
                                   )
    coughVidData.use_sampling_technique_on_dataframe()
    pred_list = []
    label_list = []

    # For each sample in the dataset get the predictions
    # and targets for each sample and
    for input, target in coughVidData:
        input.unsqueeze_(0)
        pred, label = predict(cnn, input, target, class_mapping)
        pred_list.append(pred)
        label_list.append(label)

    classification_error = np.mean(pred_list != label_list)
    f1_score = f1_score(label_list, pred_list)

    print(f"Classification Error: {classification_error}")
    print(f"F1 Score: {f1_score}")
