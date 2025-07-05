@echo off
powershell -Command "Write-Host '[2/2]' -ForegroundColor Yellow -NoNewline; Write-Host ' Cleaning static folder...'"
rmdir /s /q static
mkdir static
timeout /t 1 >nul