from blake3 import blake3


class SpikingBuffer:
    def __init__(self, buffer_size, VulkanTensor=None):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, spike):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Remove oldest spike
        self.buffer.append(spike)

    def get_buffer(self):
        return self.buffer

    def register_hook(self, hook_fn):
        """Register a hook function to be called on each new spike."""
        self.hook_fn = hook_fn


    def add_with_hook(self, spike):
        """Add spike and call hook function if registered."""
        self.add(spike)
        if hasattr(self, 'hook_fn'):
            self.hook_fn(spike)


class SpikingBufferRegistry:
    """Registry to manage multiple spiking buffers."""
    def __init__(self, name, size=None):
        self.name = name
        self.id = blake3.blake3(name.encode()).hexdigest()[:8]
        self.buffers = {}

    def _register_buffer(self, name, buffer_size, VulkanTensor=None):
        self.buffers[name] = SpikingBuffer(buffer_size, VulkanTensor)

    def _get_buffer(self, name):
        return self.buffers.get(name, None)

    def __getitem__(self, name):
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found in registry.")
        return self._get_buffer(name)
