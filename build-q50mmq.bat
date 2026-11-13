@echo off
rem S-63: build-merge with the Q5_0 MMQ instance (UD-IQ2_XS has one Q5_0 down layer; without it layer-major prompt reading is off)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PY=F:\Vs\health\AI\Strata\Strata\.venv\Scripts
"%PY%\cmake.exe" -S "%~dp0." -B "%~dp0build-merge" -DSTRATA_ORCA_Q4KS_MMQ=ON || exit /b 1
"%PY%\cmake.exe" --build "%~dp0build-merge" --target strata -j 8 || "%PY%\cmake.exe" --build "%~dp0build-merge" --target strata -j 8
