from tqdm import tqdm
import torch 
import numpy as np 

def gpu_inference_time(model, input_shape, logger=None, device=None, repetitions=100):
    """
    inference time estimation

    Args:
        model: torch model
        input_shape (list | tuple): shape of the model's batch inputs
        logger: logger. Defaults to None
        device: GPU cuda device. Defaults to None, i.e. use model's woring device
        repetitions (int, optional): testing times. Defaults to 100.
    """

    # INIT
    if device is None:
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
        else:
            raise ValueError("Please assign a GPU device for inference")
    else:
        model.to(device)

    if isinstance(input_shape, list):
        dummy_input = [torch.randn(shape_k, dtype=torch.float).to(
            device) for shape_k in input_shape]
    elif isinstance(input_shape, tuple):
        dummy_input = [torch.randn(input_shape, dtype=torch.float).to(
            device)]
    else:
        raise ValueError(
            f"`input_shape` should be a tuple or a list containing multiple tuples, but get `{input_shape}` ")

    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = model(*dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in tqdm(range(repetitions), desc='Inference Time Est:'):
            starter.record()
            _ = model(*dummy_input)
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)
    if logger:
        logger.info(
            '='*40+f'\nInference Time Estimation \nInputs Shape:\t{input_shape} \nEstimated Time:\t{mean_time:.3f}ms \nEstimated Std:\t{std_time:.3f}ms\n'+'='*40)
    else:
        print(
            '='*40+f'\nInference Time Estimation \nInputs Shape:\t{input_shape} \nEstimated Time:\t{mean_time:.3f}ms \nEstimated Std:\t{std_time:.3f}ms\n'+'='*40)
   