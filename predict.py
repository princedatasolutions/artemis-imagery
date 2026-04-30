from cog import BasePredictor, Input, Path
from transparent_background import Remover
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.remover = Remover()

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> Path:
        input_image = Image.open(image).convert("RGBA")
        output = self.remover.process(input_image)

        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)