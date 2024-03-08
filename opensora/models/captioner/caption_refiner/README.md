# Refiner for Video Caption
Transform the short caption annotations from video datasets into the long and detailed caption annotations.
* Add detailed description for background scene.
* Add detailed description for object attributes, including color, material, pose.
* Add detailed description for object-level spatial relationship.


## 🛠️ Extra Requirements and Installation
* openai == 0.28.0
* jsonlines == 4.0.0
* nltk == 3.8.1
* Install the LLaMA-Accessory:

you also need to download the weight of SPHINX to ./ckpt/ folder

## 🗝️ Refining
The refining instruction is in [demo_for_refiner.py](demo_for_refiner.py).
```bash
python demo_for_refiner.py --root_path $path_to_repo$ --api_key $openai_api_key$
```

### Refining Demos
```bash
[original caption]: A red mustang parked in a showroom with american flags hanging from the ceiling.
```

```bash
[refine caption]: In summary, A red mustang parked in a showroom with american flags hanging from the ceiling.We first describe the whole scene. This scene is most likely to take place in a car show or a car dealership. The image shows a red convertible car parked in a showroom with a large window, likely showcasing the car to potential buyers or showcasing the cars features. The presence of the car in a showroom and the large window suggests that this is a place where people come to explore and purchase cars.There are multiple objects in the video, including mustang, showroom, flags, ceiling, As for the spatial relationship. The mustang is on display in a showroom.The mustang is parked in front of flags.The mustang is parked under the ceilingThe showroom is located next to flagsThe showroom is located on the ceilingFlags hang from the ceiling
```



## 🛠️ TODO Lists
- [ ] Add GPT-3.5-Turbo for caption summarization. ⌛ [WIP]
- [ ] Add LLAVA-1.6. ⌛ [WIP]
- [ ] More descriptions. ⌛ [WIP]