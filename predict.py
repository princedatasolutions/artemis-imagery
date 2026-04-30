from cog import BasePredictor, Input, Path
from rembg import remove, new_session
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.session = None

    def predict(
        self,
        image: Path = Input(description="Input image"),
        alpha_matting: bool = Input(description="Use alpha matting for softer edges", default=True),
        threshold: float = Input(description="Hard alpha threshold. 0 keeps soft alpha.", default=0.0, ge=0.0, le=1.0),
        model: str = Input(description="rembg model/session", default="isnet-general-use"),
    ) -> Path:
        if self.session is None:
            self.session = new_session(model)

        input_image = Image.open(image).convert("RGBA")

        output = remove(
            input_image,
            session=self.session,
            alpha_matting=alpha_matting
        )

        if threshold > 0:
            alpha = output.getchannel("A")
            alpha = alpha.point(lambda p: 255 if p / 255 >= threshold else 0)
            output.putalpha(alpha)

        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)
