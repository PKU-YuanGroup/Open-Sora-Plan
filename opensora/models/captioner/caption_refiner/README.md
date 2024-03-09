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
[refine caption]: This scene depicts a red Mustang parked in a showroom with American flags hanging from the ceiling. The showroom likely serves as a space for showcasing and purchasing cars, and the Mustang is displayed prominently near the flags and ceiling. The scene also features a large window and other objects. Overall, it seems to take place in a car show or dealership.
```

- [ ] Add GPT-3.5-Turbo for caption summarization. ⌛ [WIP]
- [ ] Add LLAVA-1.6. ⌛ [WIP]
- [ ] More descriptions. ⌛ [WIP]