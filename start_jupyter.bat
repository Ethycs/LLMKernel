@echo off
REM Start Jupyter Lab with the LLM Kernel

echo Starting Jupyter Lab with LLM Kernel...

REM Enable debug logging
@REM set LLM_KERNEL_DEBUG=DEBUG
set LLM_KERNEL_DEBUG=ERROR

REM Activate pixi environment and start Jupyter
echo Activating pixi notebook environment...
pixi run -e notebook jupyter lab --no-browser --port=8888