import logging

# Required for setting SegmentDescription attributes. Direct import as this is not part of App SDK package.
from pydicom.sr.codedict import codes
from operators import PreprocessNiftiOperator, SegInferenceOperator
from monai.deploy.core import Application, resource


@resource(cpu=1, gpu=1, memory="7Gi")
class TotalBodySegmentation(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        logging.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        logging.info(f"End {self.run.__name__}")

    def compose(self):
        logging.info(self._context.model_path)
        """Creates the app specific operators and chain them up in the processing DAG."""

        logging.info(f"Begin {self.compose.__name__}")

        preprocess_op = PreprocessNiftiOperator()
        inference_op = SegInferenceOperator()
                                    
        self.add_flow(preprocess_op, inference_op, {'image': 'image', 'model': 'model'})

        logging.info(f"End {self.compose.__name__}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app_instance = TotalBodySegmentation(do_run=True)
