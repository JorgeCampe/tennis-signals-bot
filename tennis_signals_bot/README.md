# Bot de Señales — Tenis (Kalshi)

App **independiente** del dashboard Flask (localhost). Lleva una **banca simulada
que arranca en $250** (como tu bot de temperatura), apuesta las señales del modelo
contra el precio **neto** de Kalshi (comisión descontada) y **liquida** cada
apuesta con los resultados reales de ESPN. Genera su propio `dashboard.html`.

No coloca órdenes reales. Es paper trading: el juez es la banca. Si con el tiempo
no crece, no hay edge real.

## Cómo correrlo

Doble clic en **`run.bat`** (Windows), o desde una terminal en esta carpeta:

```
python bot.py           # corre la simulación y regenera dashboard.html
python bot.py --open     # además abre el dashboard en el navegador
```

Luego abre **`dashboard.html`**. Corre el bot cada cierto tiempo (o prográmalo):
en cada corrida liquida lo que ya terminó, coloca las señales nuevas y actualiza
la curva de banca.

## Qué hace en cada corrida

1. Baja resultados frescos de ESPN y **liquida** las apuestas abiertas cuyos
   partidos ya terminaron (gana / pierde / anula si nunca aparece).
2. Lee las **señales** de hoy: modelo propio (Elo v2 calibrado + ML, promediados)
   vs precio neto de Kalshi. Apuesta el lado con ventaja ≥ 5% (¼ Kelly, tope 10%
   de la banca por apuesta, cuota máx 4.0).
3. **Coloca** una apuesta por partido dentro del cash disponible.
4. Actualiza `data/positions.csv`, `data/equity.csv`, `data/signals.csv` y
   regenera `dashboard.html` (curva de banca, KPIs, Monte Carlo, tablas).

## Parámetros (editables al inicio de `bot.py`)

| Parámetro | Default | Qué es |
|---|---|---|
| `START` | 250 | banca inicial |
| `MIN_EDGE` | 0.05 | ventaja mínima del modelo vs Kalshi para apostar |
| `KELLY_FRAC` | 0.25 | fracción de Kelly (¼) |
| `MAX_STAKE_FRAC` | 0.10 | tope por apuesta (10% de la banca) |
| `MAX_ODDS` | 4.0 | cuota máxima (guarda favorito-longshot) |

## Depende de

Reutiliza el "cerebro" del proyecto NBA (por `sys.path`): `ml/kalshi.py`,
`ml/kalshi_signals.py`, `ml/tennis_models.py`, `ml/tennis_results.py`, `config.py`.
Debe vivir dentro de la carpeta `NBA` para encontrarlos. Es una app aparte (su
propia carpeta, sus datos y su interfaz), no parte del servidor Flask.

## Honestidad

El modelo **no le gana** al mercado sharp; probado sobre decenas de miles de
partidos. La apuesta a que estas señales sean rentables es que los **250 y
Challengers** en Kalshi estén mal preciados (poca liquidez, poca atención sharp).
La banca en paper lo dirá. Ventajas enormes (+30/+50%) casi siempre significan que
el modelo se equivoca, no que hay oro. Usar como paper hasta comprobar edge real.
