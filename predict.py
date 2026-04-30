from cog import BasePredictor, Input, Path
from rembg import remove
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> Path:
        input_image = Image.open(image).convert("RGBA")
        output = remove(input_image)

        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)
