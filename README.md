# ComfyUI-FastSDCPU
A set of nodes for interfacing with the FastSDCPU webserver
# Setup
1. Install the extension with dependencies
2. Install [FastSDCPU](https://github.com/rupeshs/fastsdcpu) (clone the repo, run `install.bat`/`install.sh`)
3. (Recommended) Set the environment variable `DEVICE=GPU`:

   Bash: `export DEVICE=GPU`

   Powershell: `$env:DEVICE = "GPU"`

   CMD: `set DEVICE=GPU`
5. Run `start-webserver.bat`/`start-webserver.sh`
# Converting your own models
Fastsdcpu supports regular diffusers models as well as openvino models. Openvino models are much faster than standard models on intel cpus and igpus but they also require significantly more memory. To convert them you can use the [lcm-openvino-converter](https://github.com/rupeshs/lcm-openvino-converter) repo. This repo contains commands to both apply an lcm lora and convert the model to openvino. The lcm conversion isn't required but because fastsdcpu is optimised around LCM models regular models will use inefficient sampling parameters which will require more steps than normal.

Custom models should be placed into comfyui's diffusers folder.
Selecting local models only works when the webserver is run on the same system as the comfyui instance. If the server is not local you will have to move the folder to the server and specify the folder path that's on the server.
# Previews
![preview1](https://github.com/BetaDoggo/ComfyUI-FastSDCPU/blob/main/t2i_workflow.png)
![preview2](https://github.com/BetaDoggo/ComfyUI-FastSDCPU/blob/main/i2i_workflow.png)
