from typing import Dict, Sequence, List, Union
from dataclasses import dataclass

import torch
import numpy as np
from transformers import WhisperProcessor
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS


@dataclass
class DataCollatorForLlava(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, pad_token_id, model_max_length):
        self.pad_token_id = pad_token_id
        self.model_max_length = model_max_length
        self.ignore_index = MODEL_CONSTANTS['llava']['IGNORE_INDEX']

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=self.ignore_index)
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )

        if "pixel_values" in instances[0]:
            images = [instance["pixel_values"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["pixel_values"] = torch.stack(images)
            else:
                batch["pixel_values"] = images

        return batch


class DataCollatorForInternvl(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id
        self.ignore_index = MODEL_CONSTANTS['internvl']['IGNORE_INDEX']

    def __call__(self, features):
        first = features[0]
        batch = {}

        batch_lens = [feat["input_ids"].shape for feat in features]
        max_item_length = max(batch_lens)[0]
        for feat in features:
            temp_input_ids = torch.LongTensor([self.pad_id] * max_item_length)
            temp_input_ids[:feat["input_ids"].shape[0]] = feat["input_ids"]
            feat["input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([self.ignore_index] * max_item_length)
            temp_labels[:feat["labels"].shape[0]] = feat["labels"]
            feat["labels"] = temp_labels
            feat["attention_mask"] = feat["input_ids"].ne(self.pad_id)

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let"s make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids", "pixel_values", "image_flags") and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            if k in ("pixel_values", "image_flags"):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.concat([f[k] for f in features])
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor_name_or_path, language, task):
        self.processor = WhisperProcessor.from_pretrained(
            processor_name_or_path,
            language=language,
            task=task,
        )

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


DATA_COLLATOR = {
    "llava": DataCollatorForLlava,
    "internvl":DataCollatorForInternvl,
    "whisper":DataCollatorSpeechSeq2SeqWithPadding
}
