"""
Pipeline creation and management for Vulkan compute shaders.
"""

from .base import VULKAN_AVAILABLE

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanPipelines:
    """Pipeline creation and descriptor set management"""
    
    def __init__(self, core):
        """Initialize with VulkanCore instance"""
        self.core = core
        self.pipelines = {}
        self.descriptor_set_layouts = {}
        self.pipeline_layouts = {}
        # Descriptor set caching to prevent pool exhaustion
        self.descriptor_set_cache = {}
        self.cache_access_order = []  # LRU tracking
        self.max_cache_size = 100  # Limit cache to prevent pool exhaustion
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _create_descriptor_set_layout(self, num_buffers: int):
        """Create descriptor set layout for storage buffers"""
        bindings = []
        for i in range(num_buffers):
            bindings.append(VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
            ))
        
        layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings
        )
        
        return vkCreateDescriptorSetLayout(self.core.device, layout_info, None)
    
    def _create_pipeline_layout(self, descriptor_set_layout, push_constant_size: int = 0):
        """Create pipeline layout"""
        push_constant_range = None
        if push_constant_size > 0:
            push_constant_range = VkPushConstantRange(
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size
            )
        
        layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=1 if push_constant_range else 0,
            pPushConstantRanges=[push_constant_range] if push_constant_range else None
        )
        
        return vkCreatePipelineLayout(self.core.device, layout_info, None)
    
    def _create_compute_pipeline(self, shader_code: bytes, pipeline_layout):
        """Create compute pipeline"""
        # Create shader module
        shader_info = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code
        )
        shader_module = vkCreateShaderModule(self.core.device, shader_info, None)
        
        # Create compute pipeline
        stage_info = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main"
        )
        
        pipeline_info = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=pipeline_layout
        )
        
        pipeline = vkCreateComputePipelines(self.core.device, None, 1, [pipeline_info], None)[0]
        
        # Can destroy shader module after pipeline creation
        vkDestroyShaderModule(self.core.device, shader_module, None)
        
        return pipeline
    
    def get_or_create_pipeline(self, shader_name: str, num_buffers: int, push_constant_size: int = 0):
        """
        Get or create a pipeline for a shader.
        
        Raises:
            KeyError: If shader is not found in shaders dictionary
        
        Returns:
            Tuple of (pipeline, pipeline_layout, descriptor_set_layout)
        """
        if shader_name not in self.pipelines:
            # Check if shader exists
            if shader_name not in self.core.shaders:
                raise KeyError(
                    f"Shader '{shader_name}' not found. "
                    f"Compile with: glslc shaders/{shader_name}.glsl -o shaders/spv/{shader_name}.spv"
                )
            
            desc_layout = self._create_descriptor_set_layout(num_buffers)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size)
            pipeline = self._create_compute_pipeline(
                self.core.shaders[shader_name],
                pipe_layout
            )
            
            self.descriptor_set_layouts[shader_name] = desc_layout
            self.pipeline_layouts[shader_name] = pipe_layout
            self.pipelines[shader_name] = pipeline
        
        return (
            self.pipelines[shader_name],
            self.pipeline_layouts[shader_name],
            self.descriptor_set_layouts[shader_name]
        )
    
    def _create_descriptor_set(self, layout, buffers: list):
        """Create and populate descriptor set"""
        alloc_info = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.core.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[layout]
        )
        
        descriptor_set = vkAllocateDescriptorSets(self.core.device, alloc_info)[0]
        
        # Update descriptor set with buffer bindings
        writes = []
        for i, (buffer, size) in enumerate(buffers):
            buffer_info = VkDescriptorBufferInfo(
                buffer=buffer,
                offset=0,
                range=size
            )
            
            write = VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buffer_info]
            )
            writes.append(write)
        
        vkUpdateDescriptorSets(self.core.device, len(writes), writes, 0, None)
        
        return descriptor_set
    
    def get_cached_descriptor_set(self, shader_name: str, buffers: list):
        """
        Get a cached descriptor set or create a new one.
        Implements LRU eviction to prevent descriptor pool exhaustion.
        
        Args:
            shader_name: Name of the shader
            buffers: List of (buffer, size) tuples
            
        Returns:
            Descriptor set handle
        """
        # Create cache key from shader name and buffer sizes
        buffer_sizes = tuple(size for _, size in buffers)
        cache_key = (shader_name, buffer_sizes)
        
        # Check cache
        if cache_key in self.descriptor_set_cache:
            cached_set = self.descriptor_set_cache[cache_key]
            # Update LRU order (move to end)
            if cache_key in self.cache_access_order:
                self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            
            # Update buffer bindings for the cached set
            writes = []
            for i, (buffer, size) in enumerate(buffers):
                buffer_info = VkDescriptorBufferInfo(
                    buffer=buffer,
                    offset=0,
                    range=size
                )
                write = VkWriteDescriptorSet(
                    sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=cached_set,
                    dstBinding=i,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[buffer_info]
                )
                writes.append(write)
            vkUpdateDescriptorSets(self.core.device, len(writes), writes, 0, None)
            self.cache_hits += 1
            return cached_set
        
        # Cache miss - need to create new descriptor set
        # First, evict LRU entries if cache is full
        while len(self.descriptor_set_cache) >= self.max_cache_size:
            if not self.cache_access_order:
                break
            lru_key = self.cache_access_order.pop(0)
            lru_set = self.descriptor_set_cache.pop(lru_key)
            # Free the descriptor set back to the pool
            try:
                vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [lru_set])
            except Exception as e:
                # Ignore errors during cleanup (pool might be fragmented)
                pass
        
        # Get or create descriptor set layout
        desc_layout = self.descriptor_set_layouts.get(shader_name)
        if desc_layout is None:
            # Get layout from pipeline if available
            if shader_name in self.descriptor_set_layouts:
                desc_layout = self.descriptor_set_layouts[shader_name]
            else:
                # Create layout on the fly
                desc_layout = self._create_descriptor_set_layout(len(buffers))
                self.descriptor_set_layouts[shader_name] = desc_layout
        
        # Create new descriptor set
        try:
            descriptor_set = self._create_descriptor_set(desc_layout, buffers)
        except Exception as e:
            # If allocation fails, try clearing cache and retry once
            if "OUT_OF_POOL_MEMORY" in str(e) or e == -1000069000:
                self.clear_descriptor_cache()
                descriptor_set = self._create_descriptor_set(desc_layout, buffers)
            else:
                raise
        
        # Add to cache
        self.descriptor_set_cache[cache_key] = descriptor_set
        self.cache_access_order.append(cache_key)
        self.cache_misses += 1
        
        return descriptor_set
    
    def get_cache_stats(self):
        """Get descriptor set cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_sets': len(self.descriptor_set_cache),
            'max_cache_size': self.max_cache_size,
            'max_pool_size': 500
        }
    
    def clear_descriptor_cache(self):
        """Clear all cached descriptor sets"""
        if hasattr(self.core, 'device') and self.core.device:
            for desc_set in self.descriptor_set_cache.values():
                try:
                    vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
                except Exception:
                    # Ignore errors during cleanup
                    pass
        self.descriptor_set_cache.clear()
        self.cache_access_order.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cleanup(self):
        """Cleanup pipeline resources"""
        if hasattr(self.core, 'device') and self.core.device:
            # Destroy pipelines
            for pipeline in self.pipelines.values():
                vkDestroyPipeline(self.core.device, pipeline, None)
            
            # Destroy pipeline layouts
            for layout in self.pipeline_layouts.values():
                vkDestroyPipelineLayout(self.core.device, layout, None)
            
            # Destroy descriptor set layouts
            for layout in self.descriptor_set_layouts.values():
                vkDestroyDescriptorSetLayout(self.core.device, layout, None)
