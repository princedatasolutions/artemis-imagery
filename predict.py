from cog import BasePredictor, Input, Path
from PIL import Image
import gdown
from transparent_background.Remover import Remover


_original_gdown_download = gdown.download


def _gdown_download_compat(*args, **kwargs):
    kwargs.pop("fuzzy", None)
    return _original_gdown_download(*args, **kwargs)


gdown.download = _gdown_download_compat


class Predictor(BasePredictor):
    def setup(self):
        self.remover = None

    def predict(
        self,
        image: Path = Input(description="Input image"),
        threshold: float = Input(
            description="Threshold for hard segmentation. 0 keeps soft alpha.",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        background_type: str = Input(
            description="Output type: rgba, map, green, white, blur, overlay.",
            default="rgba"
        )
    ) -> Path:
        if self.remover is None:
            self.remover = Remover()

        input_image = Image.open(image).convert("RGB")

        if threshold > 0:
            output = self.remover.process(
                input_image,
                type=background_type,
                threshold=threshold
            )
        else:
            output = self.remover.process(
                input_image,
                type=background_type
            )

        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)
