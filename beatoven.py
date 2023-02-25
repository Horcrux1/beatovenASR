import os 
import sys
import warnings
warnings.filterwarnings('ignore')
import librosa
import librosa.display as display
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchsummary import summary
from torch import optim 
from tqdm import tqdm
from torch.utils.data import DataLoader
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder, Scorer
from torchaudio.functional import edit_distance as leven_dist
from torchmetrics import WordErrorRate 

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, n_feats):
        super(ResidualCNN, self).__init__()
        self.norm = nn.LayerNorm(n_feats)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2)

    def forward(self, x):
        x = self.norm(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.conv(self.dropout(self.gelu(x))) + x


class RNN(nn.Module):
    def __init__(self, rnn_dim, hidden_size, batch_first):
        super(RNN, self).__init__()
        self.norm = nn.LayerNorm(rnn_dim)
        self.gelu = nn.GELU()
        self.gru = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, num_layers=1,
                          batch_first=batch_first, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.gru(self.gelu(self.norm(x)))
        return self.dropout(x)


class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.res_cnn = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fc = nn.Linear(n_feats * 32, rnn_dim)
        self.rnn = nn.Sequential(*[
            RNN(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2, hidden_size=rnn_dim, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.dense = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_dim, n_class)
        )
        self.alphabet = Alphabet(os.path.abspath("chars.txt"))
        self.scorer = Scorer(alphabet=self.alphabet,
                             scorer_path='librispeech.scorer', alpha=0.75, beta=1.85)

    def forward(self, x):
        x = self.res_cnn(self.cnn(x))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.permute(0, 2, 1)
        return self.dense(self.rnn(self.fc(x)))

    def beam_search_with_lm(self, xb):
        with torch.no_grad():
            out = self.forward(xb)
            softmax_out = out.softmax(2).cpu().numpy()
            char_list = []
            for i in range(softmax_out.shape[0]):
                char_list.append(ctc_beam_search_decoder(probs_seq=softmax_out[i, :], 
                                                         alphabet=self.alphabet, 
                                                         scorer=self.scorer, 
                                                         beam_size=25)[0][1])
        return char_list


def collate(data):
    spectrograms = [audio_transforms(waveform).squeeze(0).permute(1, 0)
                    for (waveform, _, utterance, _, _, _) in data]
    labels = [torch.Tensor(str_to_num(utterance.lower()))
              for (waveform, _, utterance, _, _, _) in data]
    input_lengths = [spec.shape[0] // 2 for spec in spectrograms]
    label_lengths = [len(label) for label in labels]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True) \
        .unsqueeze(1).permute(0, 1, 3, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths

# ================================================ TRAINING MODEL =====================================
def fit(model, epochs, train_data_loader, valid_data_loader):
    best_leven = 1000
    optimizer = optim.AdamW(model.parameters(), 5e-4)
    wer = WordErrorRate()
    len_train = len(train_data_loader)
    loss_func = nn.CTCLoss(blank=len(classes)).to(dev)
    for i in range(1, epochs + 1):
        # ============================================ TRAINING =======================================
        batch_n = 1
        train_levenshtein = 0
        len_levenshtein = 0
        all_train_decoded = []
        all_train_actual = []
        for _batch in tqdm(train_data_loader,
                           position=0, leave=True):
            model.train()
            spectrograms, labels, \
            input_lengths, label_lengths = _batch[0].to(dev), _batch[1].to(dev), _batch[2], _batch[3]
            optimizer.zero_grad()
            loss_func(model(spectrograms).log_softmax(2).permute(1, 0, 2),
                      labels, input_lengths, label_lengths).backward()
            optimizer.step()
            # ================================== TRAINING LEVENSHTEIN DISTANCE =========================
            if batch_n > (len_train - 5):
                model.eval()
                with torch.no_grad():
                    decoded = model.beam_search_with_lm(spectrograms)
                    for j in range(0, len(decoded)):
                        actual = num_to_str(labels.cpu().numpy()[j][:label_lengths[j]].tolist())
                        all_train_decoded.append(decoded[j])
                        all_train_actual.append(actual)
                        train_levenshtein += leven_dist(decoded[j], actual)
                        len_levenshtein += label_lengths[j]

            batch_n += 1
        # ============================================ VALIDATION ======================================
        model.eval()
        with torch.no_grad():
            val_levenshtein = 0
            target_lengths = 0
            all_valid_decoded = []
            all_valid_actual = []
            for _batch in tqdm(valid_data_loader, position=0, leave=True):
                spectrograms, labels, \
                input_lengths, label_lengths = _batch[0].to(dev), _batch[1].to(dev), _batch[2], _batch[3]
                decoded = model.beam_search_with_lm(spectrograms)
                for j in range(0, len(decoded)):
                    actual = num_to_str(labels.cpu().numpy()[j][:label_lengths[j]].tolist())
                    all_valid_decoded.append(decoded[j])
                    all_valid_actual.append(actual)
                    val_levenshtein += leven_dist(decoded[j], actual)
                    target_lengths += label_lengths[j]

        print('Epoch {}: Training Levenshtein {} | Validation Levenshtein {} '
              '| Training WER {} | Validation WER {}'
              .format(i, train_levenshtein / len_levenshtein, 
                          val_levenshtein / target_lengths,
                          wer(all_train_actual, all_train_decoded), 
                          wer(all_valid_actual, all_valid_decoded)), end='\n')
        # ============================================ SAVE MODEL ======================================
        if (val_levenshtein / target_lengths) < best_leven:
            torch.save(model.state_dict(), 
                       f=str((val_levenshtein / target_lengths) * 100).replace('.', '_') + '_' + 'model.pth')
            best_leven = val_levenshtein / target_lengths



if __name__ == "__main__":
    classes = "' abcdefghijklmnopqrstuvwxyz"
    with open("chars.txt", "w", encoding="utf-8") as fp:
        fp.write('\n'.join(list(classes)))
    fp.close()

    target_dir = "./data"
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    # train_dataset = torchaudio.datasets.LIBRISPEECH(target_dir, url="train-clean-100", download=False)
    # train_dataset = torchaudio.datasets.LIBRISPEECH(target_dir, url="train-clean-360", download=False)
    train_dataset = torchaudio.datasets.LIBRISPEECH(target_dir, url="train-other-500", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(target_dir, url="test-clean", download=False)

    audio_transforms = torchaudio.transforms.MelSpectrogram()
    num_to_char_map = {c: i for i, c in enumerate(list(classes))}
    char_to_num_map = {v: k for k, v in num_to_char_map.items()}
    str_to_num = lambda text: [num_to_char_map[c] for c in text]
    num_to_str = lambda labels: ''.join([char_to_num_map[i] for i in labels])

    train_batch_size = 32
    validation_batch_size = 32

    torch.manual_seed(7)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, collate_fn=collate, pin_memory=True)

    validation_loader = DataLoader(test_dataset, batch_size=validation_batch_size,
                                   shuffle=False, collate_fn=collate, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = SpeechRecognitionModel(n_cnn_layers=7, n_rnn_layers=5, 
                               rnn_dim=512, n_class=len(classes) + 1, n_feats=128).to(dev)

    summary(model, (1, 128, 1344))
    print("Training...")
    fit(model=model, epochs=25, train_data_loader=train_loader, valid_data_loader=validation_loader)