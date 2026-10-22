@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PY=F:\Vs\health\AI\Strata\Strata\.venv\Scripts
"%PY%\cmake.exe" -G Ninja "-DCMAKE_MAKE_PROGRAM=%PY%\ninja.exe" -S "%~dp0." -B "%~dp0build-merge" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CUDA_ARCHITECTURES=75;86;89" "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" "-DSTRATA_GGML_DIR=F:/Vs/health/AI/Strata/Strata/third_party/llama.cpp" || exit /b 1
"%PY%\cmake.exe" --build "%~dp0build-merge" --target strata -j 8 || "%PY%\cmake.exe" --build "%~dp0build-merge" --target strata -j 8
