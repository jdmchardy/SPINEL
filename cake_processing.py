"""Cake processing utilities for SPINEL.

Phase 1: load and plot 2D "cake" diffraction patterns exported as .txt files by
the Dioptas 2D-integration software (https://github.com/Dioptas/Dioptas), and
adapted from the reference tkinter app "cheesecake"
(https://gitlab.com/jdmchardy/cheesecake).

The Dioptas .txt export is a whitespace-delimited matrix:
  - Row 0        : the 2th axis. The first value is a corner placeholder and is
                   dropped, leaving the usable 2th columns.
  - Rows 1..N    : each row starts with the azimuth angle followed by the
                   intensity values across 2th.

This module keeps compute/plot helpers free of Streamlit UI logic (mirroring the
sibling PO.py module); the UI lives in spinel.py.
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.polynomial.chebyshev import Chebyshev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class CakeData:
    """Container for an imported 2D cake pattern.

    Attributes
    ----------
    twotheta : np.ndarray
        1D array of 2th values (length = number of 2th columns).
    azimuth : np.ndarray
        1D array of azimuth values (length = number of azimuth rows).
    intensity : np.ndarray
        2D intensity grid with shape (n_azimuth, n_twotheta).
    filename : str
        Source filename (for display), if known.
    """

    twotheta: np.ndarray
    azimuth: np.ndarray
    intensity: np.ndarray
    filename: str = ""


@dataclass
class CakeBackground:
    """Result of a background subtraction.

    Attributes
    ----------
    background : np.ndarray
        2D fitted background, same shape as the source intensity grid.
    subtracted : np.ndarray
        2D background-subtracted intensity (intensity - background), with values
        below the negative clip set to zero. Same shape as the source grid.
    """

    background: np.ndarray
    subtracted: np.ndarray


def _read_matrix(file):
    """Read a whitespace-delimited numeric matrix from a path or file-like object.

    Accepts either a filesystem path (str/os.PathLike) or a file-like object such
    as a Streamlit UploadedFile (whose contents may be bytes).
    """
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        # Reset the pointer so the caller can reuse the upload if needed.
        try:
            file.seek(0)
        except Exception:
            pass
        return np.loadtxt(io.StringIO(raw))
    return np.loadtxt(file)


def load_cake_data(file, filename="") -> CakeData:
    """Load a Dioptas cake .txt file into a :class:`CakeData`.

    Parameters
    ----------
    file : str | os.PathLike | file-like
        Path to the .txt file, or a file-like object (e.g. Streamlit upload).
    filename : str, optional
        Display name for the source. If not given and ``file`` is a path, the
        path itself is used.

    Returns
    -------
    CakeData

    Raises
    ------
    ValueError
        If the file is empty or does not have the expected 2D matrix layout.
    """
    if not filename and isinstance(file, str):
        filename = file

    try:
        matrix = _read_matrix(file)
    except Exception as exc:
        raise ValueError(f"Could not parse cake .txt file: {exc}") from exc

    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            "Unexpected cake file layout: expected a 2D matrix with a 2th header "
            f"row and azimuth rows, got shape {matrix.shape}."
        )

    # Row 0 is the 2th axis; the first entry is a corner placeholder and dropped.
    twotheta = matrix[0, 1:]
    # Column 0 of each subsequent row is the azimuth; the rest are intensities.
    azimuth = matrix[1:, 0]
    intensity = matrix[1:, 1:]

    return CakeData(
        twotheta=twotheta,
        azimuth=azimuth,
        intensity=intensity,
        filename=filename,
    )


def cake_to_long_dataframe(cake: CakeData) -> pd.DataFrame:
    """Return the cake as a long-form DataFrame ['2th', 'azimuth', 'Pixel Value'].

    Built vectorised (meshgrid + ravel) rather than row-by-row. This is the shape
    consumed by later processing/export phases; it is produced lazily because the
    frame can be large (n_azimuth * n_twotheta rows).
    """
    tth_grid, az_grid = np.meshgrid(cake.twotheta, cake.azimuth)
    return pd.DataFrame(
        {
            "2th": tth_grid.ravel(),
            "azimuth": az_grid.ravel(),
            "Pixel Value": cake.intensity.ravel(),
        }
    )


def background_samples(
    twotheta,
    profile,
    *,
    smoothing_sigma: float = 10.0,
    prominence_factor: float = 0.1,
    max_iter: int = 3,
    exclusion_window: int = 10,
    zero_removal_fraction: float = 0.8,
    gap_fill: bool = True,
    gap_min_width: int = 5,
    return_detail: bool = False,
):
    """Select the background-sample points for a single azimuth row.

    Adapted from cheesecake's ``sample_background_points``. The row is smoothed,
    peak regions are iteratively excluded, and zero/gap points are removed. Where
    a contiguous zero run (detector gap / beamstop) is wider than ``gap_min_width``,
    pseudo background points are injected across the gap by *linearly interpolating*
    between the surrounding real background points — this anchors the polynomial fit
    so it cannot swing wildly across the gap.

    Parameters
    ----------
    twotheta, profile : array-like
        The 2th axis and the intensity profile for one azimuth row.
    smoothing_sigma : float
        Gaussian smoothing sigma used when detecting peaks.
    prominence_factor : float
        Peak prominence as a fraction of the smoothed max.
    max_iter : int
        Number of iterative peak-exclusion passes.
    exclusion_window : int
        Number of points excluded either side of each detected peak.
    zero_removal_fraction : float
        Zero-intensity points in the first this-fraction of the 2th axis are
        dropped (low-angle beamstop region).
    gap_fill : bool
        If True, inject interpolated pseudo points across large zero gaps.
    gap_min_width : int
        Minimum contiguous zero-run width (in points) treated as a gap to fill.

    return_detail : bool
        If True, return a dict separating the real and pseudo (gap-interpolated)
        points instead of the combined tuple (used by the lineout inspector).

    Returns
    -------
    (sample_twotheta, sample_intensity) : tuple of np.ndarray
        The 2th-sorted background-sample points (real + interpolated pseudo) that
        feed the polynomial fit. If ``return_detail`` is True, instead returns a dict
        with keys ``real_tth``, ``real_I``, ``pseudo_tth``, ``pseudo_I``,
        ``valid_mask``, ``peak_tth``, ``peak_I``, ``exclusion_window`` for
        diagnostics/plotting.
    """
    twotheta = np.asarray(twotheta, dtype=float)
    profile = np.asarray(profile, dtype=float)
    n = profile.size

    smoothed = gaussian_filter1d(profile, sigma=smoothing_sigma)

    valid = np.ones(n, dtype=bool)
    # Drop zero-intensity points in the low-angle region (beamstop / gaps).
    zero_threshold = int(n * zero_removal_fraction)
    valid[:zero_threshold] &= profile[:zero_threshold] != 0

    # Iteratively reject points around detected peaks, recording the peaks found
    # (so the UI can show which zones were excluded).
    detected_peaks = []
    for _ in range(max_iter):
        valid_smoothed = smoothed[valid]
        if valid_smoothed.size == 0:
            break
        peaks, _ = find_peaks(
            valid_smoothed, prominence=prominence_factor * np.max(valid_smoothed)
        )
        if peaks.size == 0:
            break
        peak_indices = np.where(valid)[0][peaks]
        detected_peaks.extend(int(i) for i in peak_indices)
        for peak_idx in peak_indices:
            start = max(peak_idx - exclusion_window, 0)
            end = min(peak_idx + exclusion_window + 1, n)
            valid[start:end] = False
        if valid.sum() < 2 * exclusion_window:
            break

    # Identify large zero-gap runs anywhere on the axis; exclude their raw zeros
    # from the fit and remember them for pseudo-point interpolation.
    gap_runs = []
    if gap_fill:
        is_gap = profile == 0
        # Run boundaries via padded diff: edges come in (start, end) pairs.
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], is_gap.astype(np.int8), [0])))
        )
        for start, end in zip(edges[0::2], edges[1::2]):
            if (end - start) >= gap_min_width:
                gap_runs.append((start, end))
                valid[start:end] = False

    real_twotheta = twotheta[valid]
    real_intensity = profile[valid]

    # Inject interpolated pseudo points across the large gaps.
    pseudo_twotheta = np.array([], dtype=float)
    pseudo_intensity = np.array([], dtype=float)
    if gap_runs and real_twotheta.size >= 2:
        pseudo_twotheta = np.concatenate([twotheta[s:e] for s, e in gap_runs])
        pseudo_intensity = np.interp(pseudo_twotheta, real_twotheta, real_intensity)

    if return_detail:
        peak_idx = (np.unique(np.asarray(detected_peaks, dtype=int))
                    if detected_peaks else np.array([], dtype=int))
        return {
            "real_tth": real_twotheta,
            "real_I": real_intensity,
            "pseudo_tth": pseudo_twotheta,
            "pseudo_I": pseudo_intensity,
            "valid_mask": valid,
            "peak_tth": twotheta[peak_idx],
            "peak_I": profile[peak_idx],
            "exclusion_window": int(exclusion_window),
        }

    sample_twotheta = np.concatenate([real_twotheta, pseudo_twotheta])
    sample_intensity = np.concatenate([real_intensity, pseudo_intensity])
    order = np.argsort(sample_twotheta)
    return sample_twotheta[order], sample_intensity[order]


def compute_cake_background(
    cake: CakeData,
    *,
    n_bins: int = None,
    poly_degree: int = 20,
    negative_clip: float = -10.0,
    **sample_kwargs,
) -> CakeBackground:
    """Fit and subtract a per-azimuth-bin polynomial background over the cake.

    The azimuth rows are grouped into ``n_bins`` equal bins. For each bin the raw
    rows are averaged into a single binned profile, the background-sample points are
    selected with :func:`background_samples`, and a Chebyshev polynomial is fitted.
    That single bin background is then applied to *every* (finer-resolution) row in
    the bin, and subtracted. This means the user's azimuth binning directly controls
    how the background is estimated.

    Parameters
    ----------
    cake : CakeData
    n_bins : int, optional
        Number of azimuth bins. If None or >= the number of rows, each row is its
        own bin (finest resolution).
    poly_degree : int
        Chebyshev polynomial degree (clamped to ``n_samples - 1`` per bin).
    negative_clip : float
        After subtraction, values below this are set to 0 (removes large negative
        excursions from data gaps).
    **sample_kwargs
        Forwarded to :func:`background_samples` (smoothing_sigma, prominence_factor,
        max_iter, exclusion_window, zero_removal_fraction, gap_fill, gap_min_width).

    Returns
    -------
    CakeBackground
    """
    intensity = np.asarray(cake.intensity, dtype=float)
    twotheta = np.asarray(cake.twotheta, dtype=float)
    n_rows = intensity.shape[0]

    if n_bins is None or int(n_bins) >= n_rows:
        bin_index = np.arange(n_rows)
        n_eff = n_rows
    else:
        _edges, bin_index, _bw = assign_azimuth_bins(cake.azimuth, int(n_bins))
        n_eff = int(n_bins)

    background = np.zeros_like(intensity)
    for b in range(n_eff):
        rows = np.where(bin_index == b)[0]
        if rows.size == 0:
            continue
        # Average the rows in the bin, fit once, apply to all rows in the bin.
        profile = intensity[rows].mean(axis=0)
        sample_tth, sample_I = background_samples(twotheta, profile, **sample_kwargs)
        if sample_tth.size < 2:
            continue  # leave background for these rows at zero
        degree = int(min(poly_degree, sample_tth.size - 1))
        try:
            bin_background = Chebyshev.fit(sample_tth, sample_I, deg=degree)(twotheta)
        except Exception:
            bin_background = np.zeros_like(twotheta)
        background[rows] = np.nan_to_num(
            bin_background, nan=0.0, posinf=0.0, neginf=0.0
        )

    subtracted = intensity - background
    subtracted[subtracted < negative_clip] = 0.0
    return CakeBackground(background=background, subtracted=subtracted)


def assign_azimuth_bins(azimuth, n_bins: int):
    """Group azimuth rows into ``n_bins`` equal-width bins.

    Bins span the full angular coverage (each row is treated as one azimuth step
    wide), so e.g. 360 rows at 1° with 72 bins gives exactly 5.0° per bin.

    Returns
    -------
    (edges, bin_index, bin_width) :
        ``edges`` (length n_bins+1) the bin boundaries in degrees; ``bin_index``
        the 0-based bin each azimuth row falls in (same length as ``azimuth``);
        ``bin_width`` the effective bin width in degrees.
    """
    azimuth = np.asarray(azimuth, dtype=float)
    n_bins = int(n_bins)
    step = float(np.median(np.diff(azimuth))) if azimuth.size > 1 else 1.0
    step = abs(step) or 1.0
    lo = azimuth.min() - step / 2.0
    hi = azimuth.max() + step / 2.0
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_index = np.clip(np.digitize(azimuth, edges) - 1, 0, n_bins - 1)
    bin_width = (hi - lo) / n_bins
    return edges, bin_index, bin_width


def _build_heatmap(x, y, z, title: str, percentile: float) -> go.Figure:
    """Build a percentile-scaled grayscale Plotly heatmap for a 2D grid."""
    z = np.asarray(z)
    max_intensity = float(np.nanmax(z)) if z.size else 0.0
    if z.size:
        percentile = float(np.clip(percentile, 0.0, 100.0))
        zmax = float(np.nanpercentile(z, percentile))
    else:
        zmax = 0.0
    if zmax <= 0:
        zmax = max_intensity if max_intensity > 0 else 1.0

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=z,
            zmin=0,
            zmax=zmax,
            # Low intensity -> black, high intensity -> white.
            colorscale=[[0.0, "black"], [1.0, "white"]],
            colorbar=dict(title="Intensity"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="2th (degrees)",
        yaxis_title="azimuth (degrees)",
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


def plot_grid_heatmap(
    cake: CakeData, grid, title: str, percentile: float = 99.5
) -> go.Figure:
    """Plot an arbitrary 2D grid (e.g. background or subtracted) on the cake axes."""
    return _build_heatmap(cake.twotheta, cake.azimuth, grid, title, percentile)


def plot_azimuth_lineout(
    cake: CakeData,
    background: CakeBackground,
    rows,
    *,
    sample_kwargs: dict = None,
) -> go.Figure:
    """Plot the 1D lineout (2th profile) for one azimuth bin for diagnostics.

    Overlays the raw profile, the fitted background, the background-subtracted
    result, and the background-sample points (real + gap-interpolated pseudo). Each
    is a separate trace, so clicking the Plotly legend toggles it on/off. The pseudo
    and real sample points are recomputed with ``sample_kwargs`` so they match the
    parameters used to compute ``background``.

    Parameters
    ----------
    cake : CakeData
    background : CakeBackground
        Result from :func:`compute_cake_background`.
    rows : int or array-like of int
        Azimuth-row index, or the set of row indices making up an azimuth bin. When
        several rows are given the raw / background / subtracted profiles are
        averaged across the bin.
    sample_kwargs : dict, optional
        The keyword arguments passed to :func:`background_samples` when the
        background was computed (smoothing_sigma, prominence_factor, etc.), so the
        displayed sample points match the fit.
    """
    sample_kwargs = dict(sample_kwargs or {})
    sample_kwargs.pop("return_detail", None)

    rows = np.atleast_1d(np.asarray(rows, dtype=int))
    twotheta = np.asarray(cake.twotheta, dtype=float)
    profile = np.asarray(cake.intensity[rows], dtype=float).mean(axis=0)
    fitted = np.asarray(background.background[rows], dtype=float).mean(axis=0)
    subtracted = np.asarray(background.subtracted[rows], dtype=float).mean(axis=0)
    detail = background_samples(twotheta, profile, return_detail=True, **sample_kwargs)

    az_vals = np.asarray(cake.azimuth, dtype=float)[rows]
    if rows.size == 1:
        title = f"Azimuth lineout @ {az_vals[0]:.1f}°"
    else:
        title = (f"Azimuth bin {az_vals.min():.1f}° to {az_vals.max():.1f}° "
                 f"(mean of {rows.size} rows)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=twotheta, y=profile, name="Raw", mode="lines",
        line=dict(color="#888888", width=1)))
    fig.add_trace(go.Scatter(
        x=twotheta, y=fitted, name="Fitted background", mode="lines",
        line=dict(color="red", width=1.5)))
    fig.add_trace(go.Scatter(
        x=twotheta, y=subtracted, name="Background-subtracted", mode="lines",
        line=dict(color="green", width=1)))
    if detail["pseudo_tth"].size:
        fig.add_trace(go.Scatter(
            x=detail["pseudo_tth"], y=detail["pseudo_I"], name="Pseudo points (gap)",
            mode="markers", marker=dict(color="orange", size=5, symbol="circle")))
    fig.add_trace(go.Scatter(
        x=detail["real_tth"], y=detail["real_I"], name="Background samples",
        mode="markers", marker=dict(color="royalblue", size=3, symbol="x"),
        visible="legendonly"))
    if detail.get("peak_tth") is not None and detail["peak_tth"].size:
        # Detected peaks whose exclusion windows were removed from the fit; shown so
        # the user can tune the peak-search parameters.
        fig.add_trace(go.Scatter(
            x=detail["peak_tth"], y=detail["peak_I"], name="Detected peaks",
            mode="markers",
            marker=dict(color="crimson", size=9, symbol="triangle-down",
                        line=dict(width=1, color="darkred"))))

    fig.update_layout(
        title=title,
        xaxis_title="2th (degrees)",
        yaxis_title="Intensity",
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_cake_heatmap(cake: CakeData, percentile: float = 99.5) -> go.Figure:
    """Build an interactive Plotly heatmap of the imported cake.

    Parameters
    ----------
    cake : CakeData
    percentile : float
        Upper display clip set to this percentile of the pixel-intensity
        distribution (``zmax = nanpercentile(intensity, percentile)``). This
        auto-adapts per file, unlike a fixed fraction of the maximum: cake data
        is heavily skewed by a few hot pixels, so a high percentile (e.g. 99.5)
        picks a robust clip that keeps faint rings visible. Higher values darken
        the image (clip fewer bright pixels); lower values brighten it.
    """
    return _build_heatmap(
        cake.twotheta,
        cake.azimuth,
        cake.intensity,
        cake.filename or "Imported cake",
        percentile,
    )
