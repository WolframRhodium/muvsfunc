from typing import Dict
import tensorrt as trt
from utils import *


def build_engine(
    width: int,
    height: int,
    args_dict: Dict,
    max_workspace_size: int = int(1.6 * 1024 ** 3),
    logger: trt.Logger = trt.Logger(trt.Logger.VERBOSE)
) -> None:

    assert width % 2 == 0 and height % 2 == 0

    builder = trt.Builder(logger)
    builder.max_batch_size = 1

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=flags)
    network.name = "ffdnet"

    input = network.add_input(
        name="input", dtype=trt.float32, shape=(1, 3, height, width))
    sigma = network.add_input(
        name="sigma", dtype=trt.float32, shape=(1, 1, height // 2, width // 2))

    input_down = pixel_unshuffle(network, input, downscale_factor=2)
    network_input = network.add_concatenation([input_down, sigma]).get_output(0)

    conv = convolution(
        network, network_input, 13, 96, 3,
        kernel=args_dict["model.0.weight"].numpy(),
        bias=args_dict["model.0.bias"].numpy())
    relu = network.add_activation(conv, trt.ActivationType.RELU).get_output(0)

    for i in range(1, 11):
        conv = convolution(
            network, relu, 96, 96, 3,
            kernel=args_dict[f"model.{i*2}.weight"].numpy(),
            bias=args_dict[f"model.{i*2}.bias"].numpy())

        relu = network.add_activation(conv, trt.ActivationType.RELU).get_output(0)

    conv = convolution(
        network, relu, 96, 12, 3,
        kernel=args_dict[f"model.22.weight"].numpy(),
        bias=args_dict[f"model.22.bias"].numpy())

    output = pixel_shuffle(network, conv, upscale_factor=2)

    network.mark_output(output)

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    with open("timing_cache.buffer", "rb") as cache_f:
        cache = config.create_timing_cache(cache_f.read())
    config.set_timing_cache(cache=cache, ignore_mismatch=False)

    output = builder.build_serialized_network(network, config)

    with open("timing_cache.buffer", "wb") as cache_f:
        cache_f.write(cache.serialize())

    with open(f"ffdnet_{width}_{height}.engine", "wb") as f:
        f.write(output)


if __name__ == "__main__":
    import torch

    # https://github.com/HolyWu/vs-ffdnet/blob/master/vsffdnet/ffdnet_color.pth
    args_dict = torch.load("ffdnet_color.pth")

    build_engine(width=1920, height=1080, args_dict=args_dict)
