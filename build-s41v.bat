@echo off
call "%~dp0build-merge.bat" > "%~dp0build-s41v.log" 2>&1
echo exit=%errorlevel% >> "%~dp0build-s41v.log"
