import torch
from functools import reduce
from torch.nn import functional as F
from .cc_utils import check_equal
from .min_comm_cfg import min_comm_config


def extract_info_from_mm_tensors(left, right):
    m = reduce(lambda x, y: x * y, left.shape[:-1])
    k = left.shape[-1]
    check_equal(right.shape[0], k, error_info="For matmul_soc_friendly in CoC, the two input tensors left and right \
                should be of shape [..., k] and [k, n] respectively.")
    n = reduce(lambda x, y: x * y, right.shape[1:])
    return m, k, n


def is_transposed(input_):
    if input_.dim() < 2 or input_.dim() > 3:
        raise RuntimeError("input tensor of is_tensor_transposed should be either 2- or 3-dimensional")
    dim1 = input_.dim() - 1
    dim2 = input_.dim() - 2
    if input_.stride()[dim2] == 1 and input_.stride()[dim1] == reduce(lambda x, y: x * y, input_.shape[:-1]):
        return True
    else:
        return False


def ceil_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    return (a + b - 1) // b


def ceil_cc(a, b):
    if b == 0:
        raise ZeroDivisionError
    return ((a + b - 1) // b) * b


# 512B aligned shape is soc friendly
kPackage512 = 512
kPackage32 = 32


def compute_pad_num(single_dim_size, element_size, kPackage=kPackage512):
    least_size = ceil_cc(single_dim_size, ceil_div(kPackage, element_size))
    pad_num = least_size - single_dim_size
    return pad_num


# pad_dim 以 3 / 2 / 1 或者 -1, -2 / -3 的形式输入都可以
def pad_tensor(input_, pad_num, pad_dim):
    dim_size = input_.dim()
    pad_list = [0] * (dim_size * 2)
    pad_list[pad_dim * (-2) - 1] += pad_num
    input_ = F.pad(input_, tuple(pad_list), mode='constant', value=0) if pad_num > 0 else input_
    return input_


def process_with_k_aligned(left, right, mn_aligned, is_left_transposed, is_right_transposed):
    if is_left_transposed:
        left = left.contiguous()
    if not mn_aligned and not is_right_transposed:
        right = right.t().contiguous().t()
    return left, right


def process_left_with_padding_k(left, is_left_transposed, k_pad_num):
    if is_left_transposed:
        left = pad_tensor(left.permute(2, 0, 1), k_pad_num, 0)
        left = left.permute(1, 2, 0).contiguous()
    else:
        left = pad_tensor(left, k_pad_num, 2)
    return left


def process_right_with_padding_k(right, is_right_transposed, k_pad_num):
    if is_right_transposed:
        right = pad_tensor(right.t(), k_pad_num, 1)
        right = right.t()
    else:
        right = pad_tensor(right, k_pad_num, 0)
    return right


def process_with_padding_k(left, right, is_left_transposed, is_right_transposed, k_pad_num):
    left = process_left_with_padding_k(left, is_left_transposed, k_pad_num)
    right = process_right_with_padding_k(right, is_right_transposed, k_pad_num)
    return left, right


def get_aligned_mm_inputs(left, right, sp_coef=1, parallel_num=min_comm_config.parallel_num):
    """Get properly aligned tensors for matmul, according to soc friendly properties.

    Inputs
        left: the left tensor of matmul, in the shape of [m,k].
        right: the right tensor of matmul, in the shape of [k,n].
        sp_coef: the coefficient for compensating m due to any expected collective communications before the matmul.
        parallel_num: the number of parts to divide the left tensor in, by row.

    Outputs:
        left: the properly processed left tensor for matmul, in the shape of [m,k].
        right: the properly processed right tensor for matmul, in the shape of [k,n].

    """

    # The dtype of left and right tensors for matmul should be the same
    check_equal(left.element_size(), right.element_size(), error_info="In matmul_soc_friendly of CoC, the dtype of \
                left and right tensors for matmul should be the same")
    element_size = left.element_size()

    m, k, n = extract_info_from_mm_tensors(left, right)

    # check if the shape of left or right matches its memory alignment
    is_left_transposed = is_transposed(left)
    is_right_transposed = is_transposed(right)

    # After communication (if applicable) and dividing left tensor, check if m-dim and n-dim are both 512B aligned
    is_mn_aligned_512b = ((m * sp_coef // parallel_num) * element_size) % kPackage512 == 0 and (
            n * element_size) % kPackage512 == 0
    # Check if k-dim is 512B aligned
    is_k_aligned_512b = (k * element_size) % kPackage512 == 0
    # Check if k-dim is 32B aligned
    is_k_aligned_32b = (k * element_size) % kPackage32 == 0
    # Compute the required amount of padding for k-dim, if already aligned then gives 0
    k_pad_num = compute_pad_num(k, element_size, kPackage=kPackage512)

    if is_k_aligned_512b:
        return process_with_k_aligned(left, right, is_mn_aligned_512b, is_left_transposed, is_right_transposed)
    else:
        if is_mn_aligned_512b and not is_k_aligned_32b and min_comm_config.k_min <= k <= min_comm_config.k_max:
            return process_with_padding_k(left, right, is_left_transposed, is_right_transposed, k_pad_num)

    return left, right
