# Copyright 2024 HuggingFace Inc.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

import torch_npu
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.import_utils import is_torch_npu_available


def geglu_forward(self, hidden_states, *args, **kwargs):
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = ("The `scale` argument is deprecated and will be ignored. Please remove it, as passing it"
                               " will raise an error in the future. `scale` should directly be passed while calling the"
                               " underlying pipeline component i.e., via `cross_attention_kwargs`.")
        deprecate("scale", "1.0.0", deprecation_message)
    hidden_states = self.proj(hidden_states)
    if is_torch_npu_available():
        # using torch_npu.npu_geglu can run faster and save memory on NPU.
        try:
            return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]
        except NotImplementedError:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)
    else:
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)
