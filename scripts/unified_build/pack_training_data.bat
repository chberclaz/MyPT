@echo off
REM Pack all training + eval data into a single split 7z archive (300MB volumes)
REM Requires 7-Zip installed -- adjust the path below if needed.
REM Usage: run from project root (D:\coding\MyPT)

SET SEVENZIP="C:\Program Files\7-Zip\7z.exe"
SET OUTDIR=data\packed

if not exist %OUTDIR% mkdir %OUTDIR%

echo.
echo ======================================================================
echo   Packing all training data into one 7z archive (300MB volumes)
echo ======================================================================
echo.

%SEVENZIP% a -t7z -v300m -mx=5 -mmt=on "%OUTDIR%\training_data.7z" ^
    "data\unified_6B" ^
    "data\unified_phase1_circuit" ^
    "data\code_eval_tokenized" ^
    "data\retrieval_eval_tokenized"

echo.
echo ======================================================================
echo   Done! Archives in: %OUTDIR%\
echo   Upload all training_data.7z.* files to RunPod.
echo   Extract on RunPod with: 7z x training_data.7z.001
echo ======================================================================
pause
