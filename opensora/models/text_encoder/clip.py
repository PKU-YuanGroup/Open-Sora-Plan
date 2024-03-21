# -*- coding: utf-8 -*-
import os
import re
import ftfy
import torch
import html
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

class CLIPEmbedder:
    """
    A class for embedding texts and images using a pretrained CLIP model.
    """
    
    def __init__(self, device='cuda', model_name='openai/clip-vit-base-patch32', cache_dir='./cache_dir', use_text_preprocessing=True, max_length=77):
        """
        Initializes the CLIPEmbedder with specified model and configurations.
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_text_preprocessing = use_text_preprocessing
        self.max_length = max_length
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device).eval()
        
        for param in self.text_model.parameters():
            param.requires_grad = False

    def get_text_embeddings(self, texts):
        """
        Generates embeddings for a list of text prompts.
        """
        self._validate_input_list(texts, str)
        
        if self.use_text_preprocessing:
            texts = [self._clean_text(text) for text in texts]
        
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        
        return embeddings

    def encode_text(self, texts):
        """
        Encodes texts into embeddings and returns the last hidden state and pooled output.
        """
        self._validate_input_list(texts, str)
        
        batch_encoding = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length").to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(**batch_encoding)
        
        return outputs.last_hidden_state, outputs.pooler_output

    def get_image_embeddings(self, image_paths):
        """
        Generates embeddings for a list of image file paths.
        """
        self._validate_input_list(image_paths, str)
        images = [self._load_image(path) for path in image_paths]
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        
        return embeddings

    def _validate_input_list(self, input_list, expected_type):
        """
        Validates that the input is a list of expected type.
        """
        if not isinstance(input_list, list) or not all(isinstance(item, expected_type) for item in input_list):
            raise ValueError(f"Input must be a list of {expected_type.__name__}.")

    def _clean_text(self, text):
        """
        Applies basic cleaning and formatting to a text string.
        """
        text = ftfy.fix_text(text)
        text = html.unescape(text)
        return text.strip()

    def _load_image(self, image_path):
        """
        Loads and preprocesses an image from a file path.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {e}")
        return image

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls

        caption = BeautifulSoup(caption, features='html.parser').text


        caption = re.sub(r'@[\w\d]+\b', '', caption)

        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)

        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption)


        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)


        caption = re.sub(r'&quot;?', '', caption)

        caption = re.sub(r'&amp', '', caption)


        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)


        caption = re.sub(r'\d:\d\d\s+$', '', caption)


        caption = re.sub(r'\\n', ' ', caption)


        caption = re.sub(r'#\d{1,3}\b', '', caption)

        caption = re.sub(r'#\d{5,}\b', '', caption)
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  
        caption = re.sub(r'[\.]{2,}', r' ', caption)  

        caption = re.sub(self.bad_punct_regex, r' ', caption)  
        caption = re.sub(r'\s+\.\s+', r' ', caption)  
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)
        caption = self.basic_clean(caption)
        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

if __name__ == '__main__':

    clip_embedder = CLIPEmbedder()

    # Example
    text_prompts = [
        'A photo of a cute puppy playing with a ball.',
        'An image of a beautiful sunset over the ocean.',
        'A scene depicting a busy city street.'
    ]
    text_embeddings = clip_embedder.get_text_embeddings(text_prompts)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    image_paths = ['image1.jpg', 'image2.png']
    try:
        image_embeddings = clip_embedder.get_image_embeddings(image_paths)
        print(f"Image embeddings shape: {image_embeddings.shape}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

