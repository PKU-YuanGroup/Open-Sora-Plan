from dataclasses import dataclass


@dataclass
class CommonInferenceParams:
    """Inference parameters sent along with the prompts

    For an explanation of these parameters refer to this blog https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    return_log_probs: bool = False
    num_tokens_to_generate: int = 30

    def add_attributes(self, attribute_value_pair: dict):
        """Utility to add more attributes to inference params

        Use this method to pass in a custom dictonary to add more inference parameter attributes to the instance you created. Use as follows
        c = CommonInferenceParams
        c.add_attributes({'min_length':4, 'eod_id':153})

        Args:
            attribute_value_pair (dict): A dictionary containing attributes as the key names and their values as the values.
        """
        for key, value in attribute_value_pair.items():
            setattr(self, key, value)
