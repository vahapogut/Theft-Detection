@echo off
title Theft Guard AI Launcher
color 0A

echo ==================================================
echo   THEFT GUARD AI - Theft Detection System
echo   Starting up... Please wait.
echo ==================================================
echo.

:: 1. Start Backend (in new window)
echo [1/3] Starting Backend Service (Python/FastAPI)...
start "Theft Guard Backend" cmd /k "py backend.py"

:: Wait for backend to initialize
timeout /t 3 /nobreak >nul

:: 2. Start Frontend (in new window)
echo [2/3] Starting Dashboard (Next.js)...
cd dashboard
start "Theft Guard Dashboard" cmd /k "npm run dev"

:: Wait for frontend compilation
timeout /t 5 /nobreak >nul

:: 3. Open Browser
echo [3/3] Opening Browser (http://localhost:3000)...
start http://localhost:3000

echo.
echo ==================================================
echo   SYSTEM ACTIVE!
echo   To passivate, close the opened black command windows.
echo ==================================================
pause
