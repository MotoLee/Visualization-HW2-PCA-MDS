ipconfig
echo off
set /p port="Enter Port: "
python -m http.server %port%