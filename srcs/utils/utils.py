import torch
import torch.distributed as dist

def get_device_info():
    gpu_info_dict = {}
    if torch.cuda.is_available():
        gpu_info_dict["CUDA available"]=True
        gpu_num = torch.cuda.device_count()
        gpu_info_dict["GPU numbers"]=gpu_num
        infos = [{"GPU "+str(i):torch.cuda.get_device_name(i)} for i in range(gpu_num)]
        gpu_info_dict["GPU INFO"]=infos
    else:
        gpu_info_dict["CUDA_available"]=False
    return gpu_info_dict


def load_checkpoints(model,pretrained_dict,strict=False):
    # pretrained_dict = torch.load(checkpoints)
    if strict is True:
        try: 
            model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)



def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()

