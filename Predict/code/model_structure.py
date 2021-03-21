import torch
import torch.nn as nn

class simpleSpatialTimeNN(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 64):
        super(simpleSpatialTimeNN, self).__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.pool1 = nn.AdaptiveAvgPool2d((22, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 70))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(1540 * 4, n_lstm_units, 2, bidirectional=True, batch_first=True)
        # todo lstminput of shape (seq_len, batch, input_size) seq_len = lat*lon = 1540
        # input_size: The number of expected features in the input `x`
        # hidden_size: The number of features in the hidden state `h`
        # num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        #     would mean stacking two LSTMs together to form a `stacked LSTM`,
        #     with the second LSTM taking in outputs of the first LSTM and
        #     computing the final results. Default: 1
        # fixme the input shape of lstm seems to be mistaken, should b 12, 64, 6160
        #         - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
        self.linear = nn.Linear(128, 24)

    def forward(self, sst, t300, ua, va):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)

        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 1540
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)  # if flat, lstm input_dims = 1540 * 4

        x = torch.cat([sst, t300, ua, va], dim=-1)
        x = self.batch_norm(x)
        x, _ = self.lstm(x)

        x = self.pool3(x).squeeze(dim=-2)
        x = self.linear(x)
        return x

def load_model(model_dir):
    print(model_dir)
    model = simpleSpatialTimeNN()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_dir))
    else:
        model.load_state_dict(torch.load(model_dir, map_location="cpu"))
    map_location = torch.device('cpu')
    model.eval()
    return model