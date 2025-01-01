from .opensoraplan_i2v_processor import OpenSoraPlanI2VProcessor

I2V_PROCESSOR_MAPPINGS = {
    "opensoraplan_i2v_processor": OpenSoraPlanI2VProcessor
}


class I2VProcessor:
    """
    The extra processor of the image to video task
    I2VProcessor is the factory class for all i2v_processor

    Args:
        config (dict): for Instantiating an atomic methods
    """

    def __init__(self, config):
        super().__init__()
        i2v_processor_cls = I2V_PROCESSOR_MAPPINGS[config["processor_id"]]
        self.processor = i2v_processor_cls(config)

    def get_processor(self):
        return self.processor