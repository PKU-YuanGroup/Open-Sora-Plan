import logging
from collections import OrderedDict

import numpy as np

from mindspeed_mm.data.data_utils.aspect_ratio import ASPECT_RATIOS, get_closest_ratio


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            if key not in ASPECT_RATIOS:
                raise AssertionError(f"Aspect ratio {key} not found.")
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        logging.info("Number of buckets: %s", num_bucket)

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=None):
        resolution = H * W
        approx = 0.8

        fail = True
        for hw_id, t_criteria in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                if 1 in t_criteria:
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][1])
                    if rng.random() < t_criteria[1]:
                        fail = False
                        t_id = 1
                        break
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob in t_criteria.items():
                rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
                if isinstance(prob, tuple) or isinstance(prob, list):
                    prob_t = prob[1]
                    if rng.random() > prob_t:
                        continue
                if T > t_id * frame_interval and t_id != 1:
                    t_fail = False
                    break
            if t_fail:
                continue

            # leave the loop if prob is high enough
            if isinstance(prob, tuple) or isinstance(prob, list):
                prob = prob[0]
            if prob >= 1 or rng.random() < prob:
                fail = False
                break
        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

    def get_thw(self, bucket_id):
        if len(bucket_id) != 3:
            raise AssertionError

        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket
