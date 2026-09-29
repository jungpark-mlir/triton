"""hipBLASLt CPU trig_float E2M1/E8M0 input generation."""
import numpy as np
import torch


def make_hipblaslt_mxfp4_trig_matrix(rows, k):
    """Port hipBLASLt's E2M1/E8M0 ``trig_float`` CPU generator.

    hipBLASLt uses 32 independently seeded ``std::mt19937`` streams over
    contiguous scale-block ranges. Each stream draws doubles with libstdc++'s
    ``uniform_real_distribution``, takes ``cos(2*pi*u)``, truncates each value
    to an individually scaled E2M1 value, then replaces the 32 individual
    E8M0 exponents with their rounded mean and adjusts the E2M1 exponents.
    """
    if k % 32:
        raise ValueError("MXFP4 trig generation requires K divisible by 32")

    block_size = 32
    num_threads = 32
    seed = 1713573849
    blocks = rows * k // block_size
    if blocks % num_threads:
        raise ValueError(
            "MXFP4 trig generation requires scale blocks divisible by 32")

    data = torch.empty((rows, k), dtype=torch.uint8)
    scales = torch.empty((rows, k // block_size), dtype=torch.uint8)
    data_flat = data.numpy().reshape(-1)
    scale_flat = scales.numpy().reshape(-1)
    blocks_per_thread = blocks // num_threads
    blocks_per_chunk = 32 * 1024
    output_block = 0

    for thread in range(num_threads):
        rng = np.random.RandomState(seed + thread)
        remaining = blocks_per_thread
        while remaining:
            chunk_blocks = min(remaining, blocks_per_chunk)
            count = chunk_blocks * block_size

            # libstdc++ generate_canonical<double, 53> consumes two MT19937
            # words, with the first word as the low 32 bits.
            raw = rng.randint(
                0, 2**32, size=(count, 2), dtype=np.uint32)
            uniform = (
                raw[:, 0].astype(np.float64)
                + raw[:, 1].astype(np.float64) * 2.0**32) * 2.0**-64
            values = np.cos(uniform * (2.0 * np.pi))
            bits = values.view(np.uint64)

            sign = ((bits >> 63) << 3).astype(np.uint8)
            exponent = (
                ((bits >> 52) & 0x7ff).astype(np.int32) - 1023)
            mantissa = (
                (bits & ((1 << 52) - 1)) >> 51).astype(np.uint8)

            # For |cos(angle)| <= 1, the initial E2M1 exponent is zero and
            # its E8M0 scale exponent is floor(log2(abs(value))).
            element_scales = np.maximum(exponent, -127) + 127
            element_scales = element_scales.reshape(chunk_blocks, block_size)
            block_scales = np.floor(
                element_scales.mean(axis=1) + 0.5).astype(np.int32)
            adjusted = element_scales - block_scales[:, None]
            sign = sign.reshape(chunk_blocks, block_size)
            mantissa = mantissa.reshape(chunk_blocks, block_size)

            encoded = np.zeros(
                (chunk_blocks, block_size), dtype=np.uint8)
            normal = adjusted >= 0
            # Saturate overflow to finite E2M1 max (6), as mxDataGenerator does.
            encoded[normal] = sign[normal] | np.minimum(
                ((adjusted[normal] + 1) * 2 + mantissa[normal]), 7
            ).astype(np.uint8)
            subnormal = adjusted == -1
            encoded[subnormal] = sign[subnormal] | 1
            # Values below the minimum E2M1 subnormal become positive zero,
            # matching mxDataGenerator's setZero (not signed zero).

            data_begin = output_block * block_size
            data_end = data_begin + count
            data_flat[data_begin:data_end] = encoded.reshape(-1)
            scale_flat[
                output_block:output_block + chunk_blocks
            ] = block_scales.astype(np.uint8)
            output_block += chunk_blocks
            remaining -= chunk_blocks

    return data, scales

