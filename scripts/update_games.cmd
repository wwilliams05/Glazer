@echo off
rem Daily game log update for the current season.
rem Runs on this machine because stats.nba.com appears to block GitHub's servers: on 2026-11-17
rem the GitHub job spent ~5 minutes in timeouts and retries and failed, while the same command works here.
rem Output is appended to games_update.log in the repo root (ignored by git).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
echo ---- %date% %time% >> games_update.log
".venv\Scripts\python.exe" -m luka.ingest.games --current >> games_update.log 2>&1
