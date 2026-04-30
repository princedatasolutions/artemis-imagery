from cog import BasePredictor, Input, Path
from rembg import remove, new_session
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
    self.session = new_session("isnet-general-use")

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> Path:
        input_image = Image.open(image).convert("RGBA")
        output = remove(
    input_image,
    session=self.session,
    alpha_matting=True
)

        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)
