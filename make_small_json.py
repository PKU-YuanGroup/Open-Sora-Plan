import json
import random
with open('/storage/anno_jsons/stage2_pandamovie18m_m0.96_aes4.5_sucai2m_m0.95_vidal2m_allimg15m_6_8_4744757_shuffle.json', 'r') as f:
    data = json.load(f)
random.shuffle(data)
data = data[:50000]

with open('debug5000.json', 'w') as f:
    json.dump(data, f, indent=2)
