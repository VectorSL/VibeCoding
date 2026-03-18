import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extensions():
    # Get CUDA architecture from PyTorch
    cuda_arch_list = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major = torch.cuda.get_device_capability(i)[0]
            cuda_arch_list.append(f"{major}{0}")
    else:
        # Default fallback
        cuda_arch_list = ["80", "86", "89", "90"]

    # Build NVCC flags
    nvcc_flags = [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
    ]

    # Add architecture-specific flags
    arch_flags = []
    for arch in cuda_arch_list:
        arch_flags.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
    nvcc_flags.extend(arch_flags)

    # Add extra NVCC flags for optimization
    nvcc_flags.extend([
        '-maxrregcount=128',
        '--extra-device-vectorization',
    ])

    extensions = [
        CUDAExtension(
            name='flash_attention_cuda',
            sources=[
                'ccsrc/flash_attention_fwd.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_flags,
            },
        )
    ]
    return extensions


setup(
    name='flash_attention_cuda',
    version='0.1.0',
    author='FlashAttention CUDA',
    description='Hand-written CUDA implementation of FlashAttention v2',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
