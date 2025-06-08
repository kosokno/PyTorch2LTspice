@echo off
choice /c YN /m "Are you sure you want to delete all the simulation artifacts? (Y/N): "
if errorlevel 2 goto end

echo Deleting *.pth, *.sp, *.txt, and the 'gym' folder...
del /Q *.pth
del /Q *.sp
del /Q *.txt

:: Remove read-only, hidden, and system attributes in the 'gym' folder
attrib -r -h -s gym\*.* /S /D

:: Delete the 'gym' folder
rmdir /S /Q gym

echo Deletion complete.
:end
pause
