from abc import ABCMeta, abstractmethod, abstractclassmethod


class DriverBase(metaclass=ABCMeta):

    @abstractclassmethod
    def is_active(self):
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    def __init__(self) -> None:
        pass

from hip import hip
def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

class GPUDriver(DriverBase):
    stream = None
    def __init__(self):
        # TODO: support other frameworks than torch
        #import torch
        #self.get_device_capability = torch.cuda.get_device_capability
        self.get_device_capability = lambda : (9, 0)
        '''
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        '''
        self.stream = hip_check(hip.hipStreamCreate())
        self.get_ccurent_stream = self.dummy_stream
        #self.get_current_device = torch.cuda.current_device
        self.get_current_device = lambda : 0
        #self.set_current_device = torch.cuda.set_device
        self.set_current_device = self.dummy_set

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
    def dummy_set(self):
        #self.stream = hip_check(hip.hipStreamCreate())
        print("called set_device")
    def dummy_stream(self, dev_idx):
        return self.stream
