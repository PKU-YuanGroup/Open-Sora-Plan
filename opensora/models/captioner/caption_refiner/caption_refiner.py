import itertools
import numpy as np
from PIL import Image
from PIL import ImageSequence
from nltk import pos_tag, word_tokenize

from LLaMA2_Accessory.SPHINX import SPHINXModel
from gpt_combinator import caption_summary

class CaptionRefiner():
    def __init__(self, sample_num, add_detect=True, add_pos=True, add_attr=True,
                 openai_api_key=None, openai_api_base=None,
        ):
        self.sample_num = sample_num
        self.ADD_DETECTION_OBJ = add_detect
        self.ADD_POS = add_pos
        self.ADD_ATTR = add_attr
        self.openai_api_key = openai_api_key
        self.openai_api_base =openai_api_base

    def video_load_split(self, video_path=None):
        frame_img_list, sampled_img_list = [], []

        if ".gif" in video_path:
            img = Image.open(video_path) 
            # process every frame in GIF from <PIL.GifImagePlugin.GifImageFile> to <PIL.JpegImagePlugin.JpegImageFile>
            for frame in ImageSequence.Iterator(img):
                frame_np = np.array(frame.copy().convert('RGB').getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],3)
                frame_img = Image.fromarray(np.uint8(frame_np))
                frame_img_list.append(frame_img)
        elif ".mp4" in video_path:
            pass

        # sample frames from the mp4/gif
        for i in range(0, len(frame_img_list), int(len(frame_img_list)/self.sample_num)):
            sampled_img_list.append(frame_img_list[i])
        
        return sampled_img_list # [<PIL.JpegImagePlugin.JpegImageFile>, ...]

    def caption_refine(self, video_path, org_caption, model_path):
        sampled_imgs = self.video_load_split(video_path)

        model = SPHINXModel.from_pretrained(
            pretrained_path=model_path, 
            with_visual=True
        )
        
        existing_objects, scene_description = [], []
        text = word_tokenize(org_caption)
        existing_objects = [word for word,tag in pos_tag(text) if tag in ["NN", "NNS", "NNP"]]
        if self.ADD_DETECTION_OBJ:
            # Detect the objects and scene in the sampled images
    
            qas = [["Where is this scene in the picture most likely to take place?", None]]
            sc_response = model.generate_response(qas, sampled_imgs[0], max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
            scene_description.append(sc_response)

            # # Lacking accuracy
            # for img in sampled_imgs:
            #     qas = [["Please detect the objects in the image.", None]]
            #     response = model.generate_response(qas, img, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
            #     print(response)
                
        object_attrs = []
        if self.ADD_ATTR:
            # Detailed Description for all the objects in the sampled images
            for obj in existing_objects:
                obj_attr = []
                for img in sampled_imgs:
                    qas = [["Please describe the attribute of the {}, including color, position, etc".format(obj), None]]
                    response = model.generate_response(qas, img, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
                    obj_attr.append(response)
                object_attrs.append({obj : obj_attr})

        space_relations = []
        if self.ADD_POS:
            obj_pairs = list(itertools.combinations(existing_objects, 2))
            # Description for the relationship between each object in the sample images
            for obj_pair in obj_pairs:
                qas = [["What is the spatial relationship between the {} and the {}? Please describe in lease than twenty words".format(obj_pair[0], obj_pair[1]), None]]
                response = model.generate_response(qas, img, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
                space_relations.append(response)
        
        return dict(
            org_caption = org_caption,
            scene_description = scene_description,
            existing_objects = existing_objects,
            object_attrs = object_attrs,
            space_relations = space_relations,
        )
    
    def gpt_summary(self, total_captions):
        # combine all captions into a detailed long caption
        detailed_caption = ""

        if "org_caption" in total_captions.keys():
            detailed_caption += "In summary, "+ total_captions['org_caption']

        if "scene_description" in total_captions.keys():
            detailed_caption += "We first describe the whole scene. "+total_captions['scene_description'][-1]

        if "existing_objects" in total_captions.keys():
            tmp_sentence = "There are multiple objects in the video, including "
            for obj in total_captions['existing_objects']:
                tmp_sentence += obj+", "
            detailed_caption += tmp_sentence
        
        # if "object_attrs" in total_captions.keys():
        #     caption_summary(
        #         caption_list="", 
        #         api_key=self.openai_api_key, 
        #         api_base=self.openai_api_base,
        #     )
        
        if "space_relations" in total_captions.keys():
            tmp_sentence = "As for the spatial relationship. "
            for sentence in total_captions['space_relations']: tmp_sentence += sentence
            detailed_caption += tmp_sentence
        
        detailed_caption = caption_summary(detailed_caption, self.open_api_key, self.open_api_base)
        
        return detailed_caption