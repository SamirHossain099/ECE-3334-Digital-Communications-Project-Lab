@echo off
setlocal enabledelayedexpansion
set target=10.161.189.106
set count=30
set sum=0

:start
echo Pinging %target% %count% times...

REM Ping the target IP 30 times and calculate sum of response times
for /L %%i in (1,1,%count%) do (
    for /f "tokens=6 delims== " %%t in ('ping -n 1 %target% ^| find "time="') do (
        set /a sum+=%%t
    )
)

REM Calculate the average latency
set /a avg=sum/count

echo.
echo Average ping latency for %count% pings: !avg! ms

echo.
echo Press any key to rerun the test or Ctrl+C to exit.
pause >nul
goto start
