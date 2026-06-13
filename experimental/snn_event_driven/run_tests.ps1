$ErrorActionPreference = 'Continue'
$here = 'C:\Users\grill\Documents\GitHub\grilly\experimental\snn_event_driven'
Set-Location $here
$glslc = 'C:\Users\grill\VulkanSDK\Bin\glslc.exe'

Write-Output '=== TEST 1: SPIR-V compilation ==='
& $glslc -fshader-stage=compute --target-env=vulkan1.2 gif_neuron_emit.glsl -o gif_neuron_emit.spv
$e1 = $LASTEXITCODE
Write-Output ("gif_neuron_emit  exit=$e1")
& $glslc -fshader-stage=compute --target-env=vulkan1.2 synapse_scatter.glsl -o synapse_scatter.spv
$e2 = $LASTEXITCODE
Write-Output ("synapse_scatter  exit=$e2")
Get-ChildItem *.spv | Select-Object Name,Length | Format-Table -AutoSize | Out-String | Write-Output

Write-Output '=== TEST 2: NumPy reference (correctness + crossover) ==='
$py = 'C:\Users\grill\Documents\GitHub\grilly\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
& $py test_sparse_scatter_reference.py
Write-Output ("reference exit=$LASTEXITCODE")
