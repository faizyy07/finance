# quantum_price_backend.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional

# =============================================
# ========== CORE UTILITY FUNCTIONS ===========
# =============================================

def price_grid_from_ohlcv(df: pd.DataFrame, grid_width: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the price grid and extract price values from OHLCV dataframe.
    Automatically detects 'Close', 'close', or 'Adj Close' columns.
    """

    # Handle MultiIndex columns (sometimes yfinance returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Try to find a closing price column (case-insensitive)
    close_col = None
    for candidate in ["Close", "close", "Adj Close", "adjclose"]:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        raise KeyError(f"No 'Close' or similar column found. Available columns: {list(df.columns)}")

    prices = df[close_col].astype(float).values
    x = np.linspace(prices.min(), prices.max(), grid_width)
    return x, prices


# =============================================
# ========== QUANTUM WAVE SIMULATION ===========
# =============================================

def gaussian_wave_packet(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    """
    Creates a Gaussian wave packet centered at x0 with spread sigma and wavenumber k0.
    """
    norm = (1 / (sigma * np.sqrt(2 * np.pi)))
    return norm * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)


def schrodinger_step(psi: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """
    One-dimensional free Schrödinger time evolution step.
    Uses spectral (Fourier) method for numerical stability.
    """
    n = psi.size
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-1j * (k ** 2) * dt / 2.0)
    psi_next = np.fft.ifft(psi_k)
    return psi_next


def evolve_wave_packet(psi: np.ndarray, n_steps: int, dx: float, dt: float) -> np.ndarray:
    """
    Evolves the Gaussian wave packet through n_steps using Schrödinger equation.
    """
    for _ in range(n_steps):
        psi = schrodinger_step(psi, dx, dt)
    return psi


# =============================================
# ========== PROBABILITY DENSITY ===============
# =============================================

def probability_density(psi: np.ndarray) -> np.ndarray:
    """
    Compute normalized probability density |ψ|².
    """
    density = np.abs(psi) ** 2
    return density / np.trapz(density)


# =============================================
# ========== PEAK DETECTION ===================
# =============================================

def find_density_peaks(x: np.ndarray, density: np.ndarray, n_peaks: int = 3, distance: int = 10, height: float = None) -> List[Dict]:
    """
    Identify peaks in the probability density and return top n_peaks sorted by amplitude.
    Each returned dict: {'price': price_at_peak, 'prob': density_value, 'index': idx}
    """
    density = np.asarray(density).flatten()

    if height is not None:
        pk_idx, props = find_peaks(density, distance=distance, height=height)
    else:
        pk_idx, props = find_peaks(density, distance=distance)

    if pk_idx.size == 0:
        return []

    # Handle missing 'peak_heights' key
    if "peak_heights" in props:
        pk_heights = np.asarray(props["peak_heights"])
    else:
        pk_heights = density[pk_idx]

    # Sort descending by height
    order = np.argsort(pk_heights)[::-1]
    peaks = []
    for i in order[:n_peaks]:
        idx = int(pk_idx[i])
        peaks.append({
            "price": float(x[idx]),
            "prob": float(density[idx]),
            "index": idx
        })
    return peaks


# =============================================
# ========== SIGNAL INTERPRETATION ============
# =============================================

def make_signals_from_density(
    x: np.ndarray,
    density: np.ndarray,
    n_peaks: int = 3,
    min_prob: float = 1e-4,
    current_price: Optional[float] = None
) -> List[Dict]:
    """
    Produce Buy/Sell/Observe signals based on probability peaks relative to current price.
    """
    if current_price is None:
        cur_price = (x[-1] + x[0]) / 2.0
    else:
        cur_price = float(current_price)

    peaks = find_density_peaks(
        x, density, n_peaks=n_peaks, distance=max(1, int(0.01 * len(x))),
        height=min_prob if min_prob > 0 else None
    )

    signals = []
    for p in peaks:
        signal_type = "Observe"
        if p["price"] > cur_price * 1.001:
            signal_type = "Buy"
        elif p["price"] < cur_price * 0.999:
            signal_type = "Sell"
        signals.append({
            "price": p["price"],
            "prob": p["prob"],
            "signal": signal_type
        })
    return signals


# =============================================
# ========== MAIN PIPELINE FUNCTION ===========
# =============================================

def run_quantum_price_pipeline_from_df(
    df: pd.DataFrame,
    grid_width: int = 1024,
    sigma_frac: float = 0.006,
    k0: float = 0.0,
    dt: float = 0.1,
    n_steps: int = 8,
    n_peaks: int = 3
) -> Dict:
    """
    Main function to run the full quantum wave simulation from OHLCV data.
    Returns: {'x': grid, 'density': final_density, 'signals': buy/sell signals}
    """

    # Prepare grid and prices
    x, prices = price_grid_from_ohlcv(df, grid_width=grid_width)
    current_price = prices[-1]
    sigma = sigma_frac * (x.max() - x.min())

    # Initialize Gaussian wave
    psi0 = gaussian_wave_packet(x, current_price, sigma, k0)

    # Evolve wave in time
    dx = x[1] - x[0]
    psi_final = evolve_wave_packet(psi0, n_steps=n_steps, dx=dx, dt=dt)

    # Compute normalized probability density
    density = probability_density(psi_final)

    # Generate signal interpretations
    signals = make_signals_from_density(
        x, density, n_peaks=n_peaks, current_price=current_price
    )

    return {
        "x": x,
        "density": density,
        "signals": signals
    }
