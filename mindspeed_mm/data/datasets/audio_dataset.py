# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datasets import Audio, load_dataset
from torch.utils.data import Dataset
from transformers import WhisperProcessor


class AudioDataset(Dataset):
    def __init__(
        self,
        basic_param: dict,
        preprocess_param: dict,
        **kwargs,
    ):
        super().__init__()
        self.dataset = self.get_whisper_dataset(basic_param, preprocess_param)

    def get_whisper_dataset(self, basic_param, preprocess_param):
        dataset_name_or_path = basic_param.get(
            "dataset_name_or_path", "mozilla-foundation/common_voice_11_0"
        )
        language = basic_param.get("language", "hi")
        processor_name_or_path = preprocess_param.get(
            "processor_name_or_path", "openai/whisper-large-v3"
        )
        processor_language = preprocess_param.get("language", "Hindi")
        task = preprocess_param.get("task", "transcribe")
        train_dataset = load_dataset(
            dataset_name_or_path,
            language,
            split="train+validation",
            trust_remote_code=True,
        )
        train_dataset = train_dataset.remove_columns(
            [
                "accent",
                "age",
                "client_id",
                "down_votes",
                "gender",
                "locale",
                "path",
                "segment",
                "up_votes",
            ]
        )
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        processor = WhisperProcessor.from_pretrained(
            processor_name_or_path,
            language=processor_language,
            task=task,
        )
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer

        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]

            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_features[0]

            # encode target text to label ids
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            return batch

        train_dataset = train_dataset.map(prepare_dataset)
        return train_dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
