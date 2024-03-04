import sys
from collections import OrderedDict
import tensorflow_hub as hub
import torch

from src_pytorch.fvd.pytorch_i3d import InceptionI3d


def convert_name(name):
    mapping = {
        'conv_3d': 'conv3d',
        'batch_norm': 'bn',
        'w:0': 'weight',
        'b:0': 'bias',
        'moving_mean:0': 'running_mean',
        'moving_variance:0': 'running_var',
        'beta:0': 'bias'
    }

    segs = name.split('/')
    new_segs = []
    i = 0
    while i < len(segs):
        seg = segs[i]
        if 'Mixed' in seg:
            new_segs.append(seg)
        elif 'Conv' in seg and 'Mixed' not in name:
            new_segs.append(seg)
        elif 'Branch' in seg:
            branch_i = int(seg.split('_')[-1])
            i += 1
            seg = segs[i]

            # special case due to typo in original code
            if 'Mixed_5b' in name and branch_i == 2:
                if '1x1' in seg:
                    new_segs.append(f'b{branch_i}a')
                elif '3x3' in seg:
                    new_segs.append(f'b{branch_i}b')
                else:
                    raise Exception()
            # Either Conv3d_{i}a_... or Conv3d_{i}b_...
            elif 'a' in seg:
                if branch_i == 0:
                    new_segs.append('b0')
                else:
                    new_segs.append(f'b{branch_i}a')
            elif 'b' in seg:
                new_segs.append(f'b{branch_i}b')
            else:
                raise Exception
        elif seg == 'Logits':
            new_segs.append('logits')
            i += 1
        elif seg in mapping:
            new_segs.append(mapping[seg])
        else:
            raise Exception(f"No match found for seg {seg} in name {name}")

        i += 1
    return '.'.join(new_segs)

def convert_tensor(tensor):
    tensor_dim = len(tensor.shape)
    if tensor_dim == 5: # conv or bn
        if all([t == 1 for t in tensor.shape[:-1]]):
            tensor = tensor.squeeze()
        else:
            tensor = tensor.permute(4, 3, 0, 1, 2).contiguous()
    elif tensor_dim == 1: # conv bias
        pass
    else:
        raise Exception(f"Invalid shape {tensor.shape}")
    return tensor

n_class = int(sys.argv[1]) # 600 or 400
assert n_class in [400, 600]

# Converts model from https://github.com/google-research/google-research/tree/master/frechet_video_distance
# to pytorch version for loading
model_url = f"https://tfhub.dev/deepmind/i3d-kinetics-{n_class}/1"
i3d = hub.load(model_url)
name_prefix = 'RGB/inception_i3d/'

print('Creating state_dict...')
all_names = []
state_dict = OrderedDict()
for var in i3d.variables:
    name = var.name[len(name_prefix):]
    new_name = convert_name(name)
    all_names.append(new_name)

    tensor = torch.FloatTensor(var.value().numpy())
    new_tensor = convert_tensor(tensor)

    state_dict[new_name] = new_tensor

    if 'bn.bias' in new_name:
        new_name = new_name[:-4] + 'weight' # bn.weight
        new_tensor = torch.ones_like(new_tensor).float()
        state_dict[new_name] = new_tensor

print(f'Complete state_dict with {len(state_dict)} entries')

s = dict()
for i, n in enumerate(all_names):
    s[n] = s.get(n, []) + [i]

for k, v in s.items():
    if len(v) > 1:
        print('dup', k)
        for i in v:
            print('\t', i3d.variables[i].name)

print('Testing load_state_dict...')
print('Creating model...')

i3d = InceptionI3d(n_class, in_channels=3)

print('Loading state_dict...')
i3d.load_state_dict(state_dict)

print(f'Saving state_dict as fvd/i3d_pretrained_{n_class}.pt')
torch.save(state_dict, f'fvd/i3d_pretrained_{n_class}.pt')

print('Done')

