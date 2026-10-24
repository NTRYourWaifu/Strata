@echo off
call "%~dp0build-merge.bat" > "%~dp0build-h67.log" 2>&1
echo exit=%errorlevel% >> "%~dp0build-h67.log"
