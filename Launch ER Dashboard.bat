@echo off
title ER Dashboard Launcher

:: Start backend in a new window
start "ER Dashboard - Backend" cmd /k "cd /d "C:\Users\TylerGreen\Desktop\Claude Code\er-dashboard\backend" && python main.py"

:: Give the backend 3 seconds to start before opening the browser
timeout /t 3 /nobreak >nul

:: Start frontend server in a new window
start "ER Dashboard - Frontend" cmd /k "cd /d "C:\Users\TylerGreen\Desktop\Claude Code\er-dashboard\frontend" && python -m http.server 3000"

:: Give the frontend a moment then open the browser
timeout /t 2 /nobreak >nul
start "" "http://localhost:3000"
