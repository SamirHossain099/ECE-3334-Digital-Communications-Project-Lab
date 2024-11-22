@echo off
set target=192.168.137.30
set count=30

:start
set sum=0
echo Running %count% traceroutes to %target%...

for /L %%i in (1,1,%count%) do (
    for /f "tokens=2 delims==ms " %%t in ('tracert -d -h 1 %target% ^| find "ms"') do (
        set /a sum+=%%t
    )
)

set /a avg=%sum%/%count%
echo Average response time: %avg% ms

echo.
echo Press any key to rerun the test or Ctrl+C to exit.
pause >nul
goto start
