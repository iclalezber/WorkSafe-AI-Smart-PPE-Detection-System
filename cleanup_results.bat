@echo off
powershell -Command "Write-Host '[1/2]' -ForegroundColor Yellow -NoNewline; Write-Host ' Cleaning results folder...'"
del /q results\*.*
timeout /t 1 >nul