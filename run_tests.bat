@echo off
cd /d "%~dp0"
.venv\Scripts\pytest.exe tests\test_multimodal.py -v --tb=short
pause
