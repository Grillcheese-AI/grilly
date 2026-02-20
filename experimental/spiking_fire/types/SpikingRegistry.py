from abc import ABC, abstractmethod
from blake3 import blake3
from experimental.spiking_fire.utils.spiking_buffer import SpikingBuffer
import nn


class SpikingRegistry(ABC, nn.Module):

    def __init__(self):
        self.registry = {}
        self.__slots__ = ['registry']
        self.id = blake3.blake3(self.__class__.__name__.encode()).hexdigest()[:8]


    def register(self, name: str, obj):
        self.registry[name] = obj

   
    def get(self, name: str):
        return self.registry.get(name, None)
        
    
    def __getitem__(self, name):
        if name not in self.registry:
            raise KeyError(f"'{name}' not found in registry.")
        return self.get(name)
    
    def __len__(self):
        return len(self.registry)

    def __add__(self, other):
        if not isinstance(other, SpikingRegistry):
            raise ValueError("Can only add another SpikingRegistry.")
        new_registry = SpikingRegistry()
        new_registry.registry = {**self.registry, **other.registry}
        return new_registry

    def clear(self):
            self.registry.clear()



class SpikingBufferRegistry(SpikingRegistry):

    def __init__(self):
        super().__init__()

    def register_buffer(self, name: str, buffer_size, VulkanTensor=None):
        self.registry[name] = SpikingBuffer(buffer_size, VulkanTensor)

    def get(self, name: str):
        return self.registry.get(name, None)
    
    def _is_dirty(self, name: str):
        buffer = self.get(name)
        if buffer is None:
            raise KeyError(f"Buffer '{name}' not found in registry.")
        return len(buffer.get_buffer()) > 0
    
    def clear(self):
        self.registry.clear()


