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
# Previews
![preview1](https://github.com/BetaDoggo/ComfyUI-FastSDCPU/blob/main/t2i_workflow.png)
![preview2](https://github.com/BetaDoggo/ComfyUI-FastSDCPU/blob/main/i2i_workflow.png)
