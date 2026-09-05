@echo off
REM Boot check + acces LAN (0.0.0.0:7860). Voir boot_check.bat pour le detail.
REM AUCUNE authentification: reseau de confiance uniquement.
call "%~dp0boot_check.bat" --lan %*
