@echo off
REM Boot check + LAN + tunnel Cloudflare (URL publique).
REM AUCUNE authentification: l app devient joignable depuis Internet.
REM Config perso: cloudflare.local.bat (CF_TUNNEL / CF_PORT), non versionnee.
call "%~dp0boot_check.bat" --web %*
