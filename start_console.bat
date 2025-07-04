@echo off
REM Start Jupyter Console with the LLM Kernel

echo Starting Jupyter Console with LLM Kernel...

REM Reduce debug logging
set LLM_KERNEL_DEBUG=ERROR

REM Activate pixi environment and start console
echo Starting interactive console...
pixi run -e notebook jupyter console --kernel=llm_kernel