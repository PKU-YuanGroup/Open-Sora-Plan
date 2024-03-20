import torch
from transformers import CLIPProcessor, CLIPTextModel, CLIPTokenizer
from huggingface_hub import snapshot_download


class ClipEncoder:
    """
    Embeds text prompt into vector representations. Also handles text dropout for classifier-free guidance.
    """

    def __init__(
            self,
            from_pretrained="openai/clip-vit-large-patch14",
            cache_dir='./cache_dir',
            model_max_length=77,
            device="cuda",  
            torch_type=torch.float32,  
    ):
        super().__init__()
        assert from_pretrained is not None, "Please specify the path to the T5 model"
        # cache_dir = os.path.join(cache_dir, 'clip')
        snapshot_download(repo_id=from_pretrained)
        self.device = torch.device(device)

        self.processor = CLIPProcessor.from_pretrained(from_pretrained, cache_dir=cache_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(from_pretrained, cache_dir=cache_dir)
        self.transformer = CLIPTextModel.from_pretrained(from_pretrained, cache_dir=cache_dir).to(self.device, dtype=torch_type)

        self.torch_type = torch_type
        self.model_max_length = model_max_length
        self.output_dim = self.transformer.config.hidden_size
        self.to(device)


    def _freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_text_embeddings(self, text):
        self.transformer.eval()

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        print(self.device)
        with torch.no_grad():
            outputs = self.transformer(input_ids=tokens)
            text_encoder_embs = outputs.last_hidden_state.to(self.device)
            text_tokens_and_mask = batch_encoding["attention_mask"].to(self.device)

        return text_encoder_embs, text_tokens_and_mask

    def get_embeddings(self, text=None, images=None):
        self.transformer.eval()

        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
            max_length=self.model_max_length,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.transformer(**inputs)
            text_encoder_embs = outputs.last_hidden_state.to(self.device)

        return text_encoder_embs

    def null(self, n):
        null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y

    def to(self, dtype):
        self.transformer = self.transformer.to(dtype)
        return self


if __name__ == '__main__':
    clip = ClipEncoder(device="cuda:0", torch_type=torch.float16)  
    prompts = ['I am a test caption', 'Test twice']
    with torch.no_grad():
        print(clip.get_text_embeddings(prompts))
