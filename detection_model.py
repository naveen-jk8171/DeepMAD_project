import torch
import torch.nn as nn

class DeepMADDetectionNet(nn.Module):
    def __init__(self):
        super(DeepMADDetectionNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(num_features=128, momentum=0.8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(num_features=256, momentum=0.8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(num_features=512, momentum=0.8)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(num_features=512, momentum=0.8)
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(in_features=512 * 4, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    print("Initializing DeepMAD Detection Network...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DeepMADDetectionNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model successfully built with {total_params:,} trainable parameters.\n")
    print("Generating dummy magnetic anomaly data (Batch of 5, 128 samples each)...")
    dummy_input = torch.randn(5, 1, 128).to(device)
    print("Running forward pass...")
    output = model(dummy_input)
    print("\nOutput Probabilities (1 = Target Exists, 0 = Only Noise):")
    print(output.detach().cpu().numpy())