import json

json_path = '/work/share1/caption/osp/0410/random_video_final_1_3006269.json'
save_json_path = '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/test_100k.json'

with open(json_path, 'r') as f:
    json_data = json.load(f)

sample_num = 100000
json_data = json_data[:sample_num]

with open(save_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

