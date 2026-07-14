"""Config minimo para el bot de senales de tenis (Kalshi).

NO contiene ninguna API key. El bot solo usa datos PUBLICOS (Kalshi y ESPN),
por eso es seguro tener este repo publico.
"""
import os

MAX_ODDS_SIM = float(os.getenv("MAX_ODDS_SIM", "4.0"))
KALSHI_FEE_RATE = float(os.getenv("KALSHI_FEE_RATE", "0.07"))
KALSHI_ENABLED = os.getenv("KALSHI_ENABLED", "1") == "1"
KALSHI_BANKROLL = float(os.getenv("KALSHI_BANKROLL", "250"))
KALSHI_SIGNAL_MIN_EDGE = float(os.getenv("KALSHI_SIGNAL_MIN_EDGE", "0.05"))
