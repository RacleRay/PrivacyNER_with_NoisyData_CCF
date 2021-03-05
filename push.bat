@echo off
echo -------Begin-------
git status
set  /p  msg=Input commit:
git add .
git commit -m %msg%
git pull
git push
echo Push success: [%msg%]
echo --------End!--------
pause