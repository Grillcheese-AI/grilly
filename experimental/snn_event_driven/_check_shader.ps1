$ErrorActionPreference = 'Continue'
$glslang = 'C:\Users\grill\VulkanSDK\Bin\glslangValidator.exe'
$src = 'C:\Users\grill\Documents\GitHub\grilly\shaders\spike-scatter.glsl'
$dst = 'C:\Users\grill\Documents\GitHub\grilly\shaders\spv\spike-scatter.spv'
& $glslang -V -S comp $src -o $dst --target-env vulkan1.3
$code = $LASTEXITCODE
Write-Output ("glslangValidator exit=$code")
if (Test-Path $dst) {
    $len = (Get-Item $dst).Length
    Write-Output ("spv written: $len bytes")
} else {
    Write-Output "spv NOT written"
}
