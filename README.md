# Bot de Señales de Tenis (Kalshi) — corre solo en la nube

Corre en **GitHub Actions** cada 3 horas, **sin tu PC**. Lleva una banca simulada
(paper) que arranca en **$250**, apuesta las señales del modelo contra el precio
neto de Kalshi y liquida con resultados de ESPN. Publica un **dashboard** que ves
desde el celular vía GitHub Pages.

No usa ninguna API de pago: Kalshi y ESPN son públicos. Por eso este repo no
contiene ninguna key y es seguro tenerlo público.

## Puesta en marcha (una sola vez, ~5 min)

1. En GitHub, crea un repo **nuevo y PÚBLICO** llamado `tennis-signals-bot`
   (sin README, sin .gitignore).
2. Desde esta carpeta, en una terminal:

   ```bash
   git init
   git add .
   git commit -m "bot inicial"
   git branch -M main
   git remote add origin https://github.com/JorgeCampe/tennis-signals-bot.git
   git push -u origin main
   ```

3. En el repo: **Settings → Pages → Source: Deploy from a branch →
   Branch: `main` / carpeta `/docs` → Save**. En 1 min tendrás el dashboard en
   `https://jorgecampe.github.io/tennis-signals-bot/`.
4. Pestaña **Actions**: si pide habilitar workflows, acepta. Entra a
   *Tennis Signals Bot* y toca **Run workflow** para correrlo ya (o espera al
   próximo bloque de 3 horas).

Listo. De ahí en adelante corre solo, guarda el historial y actualiza el
dashboard cada 3 horas.

## Cambiar la frecuencia

Edita el `cron` en `.github/workflows/tennis-bot.yml`. Ejemplos:
`0 */6 * * *` (cada 6 h), `0 12 * * *` (una vez al día a las 12:00 UTC).

## Correrlo en tu PC también

`python tennis_signals_bot/bot.py --open` (abre el dashboard local). Es el mismo
bot; el estado en la nube y el local son independientes.

## Seguridad

Este repo no tiene keys. **Aparte:** tu repo `nba-betting` sí tiene la API key de
The Odds API commiteada en `config.py`. Si ese repo es público, rótala en
the-odds-api.com y quítala del historial. Este bot no la necesita.
