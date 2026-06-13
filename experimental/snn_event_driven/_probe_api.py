import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import grilly_core as g

print("MODULE_OK")
print("module attrs (sample):", [a for a in dir(g) if not a.startswith("__")][:40])
print("has create_backend:", hasattr(g, "create_backend"))
print("has Device:", hasattr(g, "Device"))

b = g.create_backend("vulkan")
print("DEVICE:", b.device_name())
meths = [m for m in dir(b) if not m.startswith("__")]
print("BACKEND_METHODS:", meths)
need = ("dispatch", "create_buffer", "upload", "download", "load_shader",
        "begin_batch", "end_batch", "barrier")
print("low-level present:", {x: hasattr(b, x) for x in need})

# Device class (the other binding path)
if hasattr(g, "Device"):
    d = g.Device()
    dm = [m for m in dir(d) if not m.startswith("__")]
    print("DEVICE_CLASS_METHODS:", dm)
