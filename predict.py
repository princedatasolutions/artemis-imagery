from cog import BasePredictor, Input, Path
from PIL import Image, ImageFilter
import torch
import numpy as np
from transformers import AutoModelForImageSegmentation
from torchvision import transforms


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(
        self,
        image: Path = Input(description="Input image"),
        edge_cleanup: int = Input(
            description="Shrink alpha edge slightly. 0 = none, 3 = light cleanup, 5 = stronger cleanup.",
            default=0,
            ge=0,
            le=9
        ),
        feather: float = Input(
            description="Soft feather after cleanup. 0 = none.",
            default=0.0,
            ge=0.0,
            le=5.0
        )
    ) -> Path:
        input_image = Image.open(image).convert("RGB")
        original_size = input_image.size

        tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
    preds = self.model(tensor)

    # BiRefNet returns dict with 'pred'
    if isinstance(preds, dict) and "pred" in preds:
        pred = preds["pred"]
    else:
        pred = preds

    pred = torch.sigmoid(pred)
    pred = pred[0][0].detach().cpu().numpy()

        pred = (pred * 255).astype(np.uint8)
        mask = Image.fromarray(pred).resize(original_size, Image.LANCZOS)

        if edge_cleanup > 0:
            if edge_cleanup % 2 == 0:
                edge_cleanup += 1
            mask = mask.filter(ImageFilter.MinFilter(edge_cleanup))

        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

        output_image = input_image.convert("RGBA")
        output_image.putalpha(mask)

        output_path = "/tmp/output.png"
        output_image.save(output_path)

        return Path(output_path)
