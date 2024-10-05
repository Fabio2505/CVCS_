import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from PIL import Image
import torchvision.transforms as transforms


def compute_depth_values(img_depth, channel):
    """
    Calcola la distanza media, il valore massimo e il minimo basati sulla profondità predetta,
    ignorando i valori di profondità pari a 0.
    """
    depth_map = img_depth[:, channel, :, :]
    valid_depths = depth_map[depth_map != 0]

    if valid_depths.numel() == 0:  # Controlla se ci sono valori validi
        return {
            "mean_depth": torch.tensor(0.0),
            "max_depth": torch.tensor(0.0),
            "min_depth": torch.tensor(0.0)
        }

    mean_depth = torch.mean(valid_depths)
    max_depth = torch.max(valid_depths)
    min_depth = torch.min(valid_depths)

    # Restituisce un dizionario con i tre valori
    return {
        "mean_depth": mean_depth,
        "max_depth": max_depth,
        "min_depth": min_depth
    }


def depth_img_preprocessing(img_path, dim):
    """
    pre-processing dell'immagine: prende la metà superiore dell'immagine e la ridimensiona per input modello ANSRGB
    """
    img = Image.open(img_path)

    width, height = img.size
    box = (0, 0, width, height // 2)  # Ritaglia la metà superiore
    img_crop = img.crop(box)
    # img_crop.show()
    img_out = img_crop.resize((dim[1], dim[2]))
    img_out.show()

    #  trasformazioni per convertire l'immagine in un tensor e normalizzarla
    preprocess_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Valori di normalizzazione
    ])

    # Applica le trasformazioni all'immagine ridimensionata
    image_tensor = preprocess_to_tensor(img_out).unsqueeze(0)  # Aggiungo la dimensione batch ->(1, 3, 128, 128)

    return image_tensor


def softmax_2d(x):
    b, h, w = x.shape
    x_out = F.softmax(x.view(b, h * w), dim=1)
    x_out = x_out.view(b, h, w)
    return x_out


# ================================ Anticipation base ==================================


class BaseModel(nn.Module):
    def __init__(self, normalize_channel_0="sigmoid", normalize_channel_1="softmax"):
        super().__init__()
        # Set normalization functions
        if normalize_channel_0 == "sigmoid":
            self.normalize_channel_0 = torch.sigmoid
        elif normalize_channel_0 == "softmax":
            self.normalize_channel_0 = softmax_0d  # manca la definizione di softmax_0d

        if normalize_channel_1 == "sigmoid":
            self.normalize_channel_1 = torch.sigmoid
        elif normalize_channel_1 == "softmax":
            self.normalize_channel_1 = softmax_2d
        """
        self.config = cfg

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "sigmoid":
            self.normalize_channel_0 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "softmax":
            self.normalize_channel_0 = softmax_0d  # manca la definizione di softmax_0d

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "sigmoid":
            self.normalize_channel_1 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "softmax":
            self.normalize_channel_1 = softmax_2d
        """

        self._create_gp_models()

    def forward(self, x):
        final_outputs = {}
        gp_outputs = self._do_gp_anticipation(x)
        final_outputs.update(gp_outputs)

        return final_outputs

    def _create_gp_models(self):
        raise NotImplementedError

    def _do_gp_anticipation(self, x):
        raise NotImplementedError

    def _normalize_decoder_output(self, x_dec):
        x_dec_c0 = self.normalize_channel_0(x_dec[:, 0])
        x_dec_c1 = self.normalize_channel_1(x_dec[:, 1])
        return torch.stack([x_dec_c0, x_dec_c1], dim=1)


# ============================= Anticipation models ===================================


class ANSRGB(BaseModel):
    """
    Predicts depth-projection from RGB only.
    """

    def _create_gp_models(self):
        resnet = tmodels.resnet18(pretrained=True)
        self.main = nn.Sequential(  # (3, 128, 128)
            # Feature extraction
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,  # (512, 4, 4)
            # FC layers equivalent
            nn.Conv2d(512, 512, 1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Upsampling
            nn.Conv2d(512, 256, 3, padding=1),  # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (256, 8, 8)
            nn.Conv2d(256, 128, 3, padding=1),  # (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (128, 16, 16),
            nn.Conv2d(128, 64, 3, padding=1),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (64, 32, 32),
            nn.Conv2d(64, 32, 3, padding=1),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (32, 64, 64),
            nn.Conv2d(32, 2, 3, padding=1),  # (2, 64, 64)
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (2, 128, 128),
        )

    def _do_gp_anticipation(self, x):
        x_dec = self.main(x["rgb"])
        x_dec = self._normalize_decoder_output(x_dec)
        outputs = {"occ_estimate": x_dec}

        return outputs


