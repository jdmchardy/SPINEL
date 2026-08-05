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
from plotly.subplots import make_subplots
from PIL import Image
from numpy.polynomial.chebyshev import Chebyshev
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, fftconvolve


# User-facing documentation rendered in the app's "How the background subtraction
# works" panel. Kept here so the behaviour and its description live together.
BACKGROUND_HELP_MD = """\
### How the background subtraction works

The experimental cake is background-subtracted **one azimuth bin at a time**, and the
azimuth binning you choose *drives* the fit:

1. **Bin** the azimuth rows into *N* equal bins — control **Number of azimuth bins**.
2. **Average** the raw rows in each bin into a single profile.
3. **Fit** a smooth polynomial background to that binned profile (two-stage peak
   search, below).
4. **Apply** that bin's background to *every* (finer-resolution) row in the bin and
   subtract it.

Fewer bins → a cleaner profile to fit but coarser azimuthal detail; more (finer) bins
usually track azimuthal variation better. The default is half the number of azimuth
rows, and the maximum is one bin per data row.

#### Fitting one binned profile

Background points are chosen by finding and excluding the diffraction peaks, then a
**Chebyshev polynomial** (**Polynomial degree**) is fit to what remains. Peaks are
found in **two stages**, both governed by **Prominence factor** — the fraction of the
current signal maximum a feature must exceed to count as a peak. *Set it above 1 to
select no peaks at all.*

**Stage 1 — Pre-fit peak search** (**Peak-search iterations**)
- Runs on the *smoothed* raw profile (**Smoothing σ** sets the smoothing width, used
  for detection only — it does not smooth the fitted background).
- **One pass is always done**; the control is the number of *additional* passes.
- Each pass detects peaks and masks ±**Peak exclusion window** points around each.
- After the first pass the biggest peaks are excluded, so the running maximum drops and
  each further pass catches progressively *weaker* peaks.
- Peaks are **masked, not subtracted** — the data is unchanged; only the set of points
  used for the fit shrinks. An initial background is then fit.

**Stage 2 — Residual refinement** (**Refinement iterations**)
- The fitted **background is subtracted**, and the *residual* is searched (only within
  the current background regions) for peaks the pre-fit stage missed — shallow peaks
  sitting on a sloped background.
- A missed peak must clear both the prominence threshold **and** a noise floor
  (5× the robust noise level), so noise is not mistaken for peaks.
- Candidates sitting right next to an already-excluded peak are **discarded as
  artifacts** — imperfect background subtraction leaves small lobes at peak edges.
- Each accepted peak is excluded and the background refit; the loop stops once a pass
  finds nothing new.

Both counts are *additional* passes: **Refinement iterations = 0** does the initial fit
only; **Peak-search iterations = 0** still runs its single baseline pass (raise
**Prominence factor** above 1 to exclude no peaks at all).

#### Data gaps (detector gaps / beamstop)

Contiguous runs of zero intensity wider than **Gap min width** are treated as gaps.
A polynomial can swing wildly across an empty gap, so **pseudo background points** are
inserted across each gap by linear interpolation from the real background on either
side (**Gap fill**). Because intensity tapers toward zero at the gap edges, the
interpolation anchors are taken from **Gap edge pad** points *beyond* the gap, so the
pseudo points sit at the true baseline instead of the weak tapered values.

- **Negative clip** — after subtraction, values below this are set to 0, removing large
  negative dips where the fit overshoots a gap.

#### Inspecting the result

- The **Fitted background** and **Background-subtracted** images show the 2D result.
- Use the **Azimuth for lineout** slider to pick a bin; it is highlighted as a
  translucent band on both images. The slider reads in **degrees** (the azimuthal
  angle, so a full ring runs roughly −180° → +180°) and moves **one bin per step**, so
  its step size is the bin width and follows your bin count.
- The **Lineout inspector** plots that bin's averaged raw profile, fitted background,
  subtracted result, the background sample points, the gap pseudo-points, and the
  **detected peaks** (red ▼) — so you can see exactly which peaks were found and which
  zones were excluded, and tune the parameters accordingly. Toggle any trace via the
  plot legend.

#### Exporting & reusing backgrounds

- **Download** the fitted background or the background-subtracted cake as a
  Dioptas-format **`.txt`** (re-loadable here) or a 32-bit float **`.tiff`**. Press
  **Prepare download files** first — the files are built for the *current* result and
  are invalidated whenever you recompute or reload, so you can never download an
  outdated cake.
- **Load a pre-made background** (a `.txt` or `.tiff` matching the cake's size) to
  subtract it directly instead of fitting a new one.
"""


# User-facing documentation for the Peak Extraction tool.
PEAK_EXTRACTION_HELP_MD = """\
### How the peak search works

Peak extraction turns a background-subtracted cake into **(azimuth, 2θ) points grouped by
hkl reflection** — the experimental input for 2D strain refinement. Everything happens
inside your chosen 2θ window, in three stages.

#### 1 · Seed the rings (once, globally)
The **azimuth max-projection** is formed — the maximum intensity at each 2θ across *all*
azimuths — so a ring that appears at only a few azimuths (a texture arc) still shows up.
Its peaks are found, and the strongest **Max hkl peaks (N)** become the initial ring
centres.
- **Seed sensitivity** — a feature must exceed this fraction of the projection's maximum
  to be seeded. *Lower seeds fainter rings.* Typical **0.01–0.1** (default 0.03); raise it
  if noise/spurious rings get seeded, lower it if a real weak ring is missed.
- **Min seed spacing** — the smallest 2θ gap allowed between two seeds, so one broad ring
  isn't seeded twice. Set it a bit below your closest real ring spacing. Typical
  **0.3–1.0°**.

#### 2 · Track the rings across azimuth
Walking the azimuth bins in order, a *running centre* is kept per ring. In each bin (the
averaged lineout of its rows) a window around the ring's running centre is searched for
the local maximum; if it clears a noise floor the ring is recorded there and its running
centre moves to the new position — so it follows strain-induced 2θ shifts and couples each
bin to the ones before it. A ring with no qualifying peak in a bin keeps its centre and
can resume later, which is how **azimuth gaps from preferred orientation** are handled.
- **Detection σ** — a peak counts only if it rises this many robust standard deviations
  above the window's noise. Typical **3–8** (default 5); lower catches fainter arcs but
  more noise, higher keeps only strong ring segments. A *noise* floor (not a fraction of
  each ring's own height) is used so a strong ring never hides a weak one.
- **Ring-track tolerance** — how far (in 2θ) a peak may sit from a ring's running centre
  to still be attached to it. Typical **0.2–1.0°**; `0` = auto (~0.4× the seed spacing).
  Larger tolerates bigger strain shifts but risks jumping to a neighbouring ring.

#### 3 · Fit each detected peak
Each detected peak is refined with the chosen **Peak shape** over a small local window of
2θ samples centred on the located point, giving a sub-bin 2θ centre, amplitude and FWHM (and,
for Pseudo-Voigt, the fitted Gaussian↔Lorentzian fraction *gl*). If the fit fails or its
centre leaves the window, the raw located position is kept.
- **Fit window** — the half-width, in 2θ *samples*, of the local slice fitted around each
  peak: the fit spans ±this many points (2·N+1 samples total). It should comfortably cover a
  peak's full width without reaching into its neighbours — too small and the shape/FWHM are
  poorly constrained, too large and an adjacent ring or curved residual biases the fit.
  Typical **8–25 samples**; the control shows the matching ±° for your data's 2θ step.
- **Peak shape** — Pseudo-Voigt (a *gl*-weighted blend of a Gaussian and a Lorentzian of the
  same width, FWHM = 2·σ; *gl* is fitted per peak from a 0.5 start and reported in the table)
  suits most powder peaks. Gaussian is for clean, symmetric peaks and reports no *gl*.
- **Azimuth bins** — how many bins the cake is averaged into for the search. More bins =
  finer azimuthal detail but noisier lineouts; usually match the background binning.

#### Result & correction
One approximate 2θ per (azimuth bin, ring) is produced, shown overlaid on the cake coloured
by group and listed in the editable table (ordered by group → azimuth → 2θ). Fix mistakes
by editing a point's **group** or deleting outlier rows — overlapping/crossing rings are
the usual thing to correct by hand.
"""


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


def _find_gap_runs(profile, gap_fill, gap_min_width):
    """Return (start, end) index ranges of contiguous zero runs >= gap_min_width."""
    gap_runs = []
    if gap_fill:
        is_gap = profile == 0
        # Run boundaries via padded diff: edges come in (start, end) pairs.
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], is_gap.astype(np.int8), [0])))
        )
        for start, end in zip(edges[0::2], edges[1::2]):
            if (end - start) >= gap_min_width:
                gap_runs.append((int(start), int(end)))
    return gap_runs


def _pad_and_merge_runs(runs, pad, n):
    """Widen each (start, end) run by ``pad`` points on each side and merge overlaps.

    Padding pushes the interpolation anchors out past the intensity taper that
    surrounds a zero gap, so the pseudo points reflect the true baseline rather than
    the (weaker) tapered edge values.
    """
    if not runs:
        return []
    padded = sorted((max(int(s) - pad, 0), min(int(e) + pad, n)) for s, e in runs)
    merged = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _select_background_samples(twotheta, profile, base_valid, peak_indices,
                               exclusion_window, gap_runs):
    """Build background-sample points given the peaks to exclude.

    Starts from ``base_valid`` (leading zeros + large gaps already removed), excludes
    an exclusion window around each peak, then injects linearly-interpolated pseudo
    points across the gap runs. Returns
    ``(valid, real_tth, real_I, pseudo_tth, pseudo_I, sample_tth, sample_I)``.
    """
    n = profile.size
    valid = base_valid.copy()
    for p in peak_indices:
        start = max(int(p) - exclusion_window, 0)
        end = min(int(p) + exclusion_window + 1, n)
        valid[start:end] = False

    real_tth = twotheta[valid]
    real_I = profile[valid]

    pseudo_tth = np.array([], dtype=float)
    pseudo_I = np.array([], dtype=float)
    if gap_runs and real_tth.size >= 2:
        pseudo_tth = np.concatenate([twotheta[s:e] for s, e in gap_runs])
        pseudo_I = np.interp(pseudo_tth, real_tth, real_I)

    sample_tth = np.concatenate([real_tth, pseudo_tth])
    sample_I = np.concatenate([real_I, pseudo_I])
    order = np.argsort(sample_tth)
    return valid, real_tth, real_I, pseudo_tth, pseudo_I, sample_tth[order], sample_I[order]


def _fit_polynomial(twotheta, sample_tth, sample_I, poly_degree):
    """Fit a Chebyshev polynomial to the sample points, evaluated over ``twotheta``."""
    if sample_tth.size < 2:
        return np.zeros_like(twotheta)
    degree = int(min(poly_degree, sample_tth.size - 1))
    try:
        background = Chebyshev.fit(sample_tth, sample_I, deg=degree)(twotheta)
    except Exception:
        background = np.zeros_like(twotheta)
    return np.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0)


def fit_bin_background(
    twotheta,
    profile,
    *,
    poly_degree: int = 45,
    smoothing_sigma: float = 10.0,
    prominence_factor: float = 0.25,
    peak_iterations: int = 1,
    iterations: int = 1,
    exclusion_window: int = 25,
    gap_fill: bool = True,
    gap_min_width: int = 5,
    gap_pad: int = 10,
    return_detail: bool = False,
):
    """Fit the polynomial background of one (binned) azimuth profile.

    Procedure:
      1. Smooth the profile (for peak detection only) and mark leading zeros and
         large detector gaps as non-background.
      2. **Pre-fit peak-search loop** (one pass always, plus ``peak_iterations``
         additional passes): detect peaks on the smoothed profile and exclude a window
         around each. Each pass excludes the peaks found so far, lowering the running
         maximum so progressively weaker peaks are caught. Then fit an initial Chebyshev
         background.
      3. **Residual refinement loop** (run ``iterations`` times, after the fit):
         subtract the current background and search the residual *within the current
         background regions* for peaks the pre-fit passes missed (shallow peaks sitting
         on the background); add them to the exclusion set and refit. Stops early once
         a pass finds no new peaks.

    Both loops are gated by ``prominence_factor``, so a large value (> 1) selects no
    peaks in either stage.

    Large gaps are bridged with linearly-interpolated pseudo points so the polynomial
    stays anchored across them. Peaks are always searched on the full 2th array (not a
    compressed valid-only array), which avoids spurious peaks at exclusion joins.

    Returns the fitted background evaluated over ``twotheta``. If ``return_detail`` is
    True, returns a dict with keys ``background``, ``real_tth``, ``real_I``,
    ``pseudo_tth``, ``pseudo_I``, ``valid_mask``, ``peak_tth``, ``peak_I``,
    ``exclusion_window`` for diagnostics/plotting.
    """
    twotheta = np.asarray(twotheta, dtype=float)
    profile = np.asarray(profile, dtype=float)
    n = profile.size
    smoothed = (gaussian_filter1d(profile, sigma=smoothing_sigma)
                if smoothing_sigma > 0 else profile)

    # Base validity: large zero gaps are removed (and later bridged with pseudo points).
    base_valid = np.ones(n, dtype=bool)
    gap_runs = _find_gap_runs(profile, gap_fill, gap_min_width)
    # Pad each gap so the interpolation anchors sit beyond the intensity taper at the
    # gap edges (otherwise the pseudo points are anchored to near-zero tapered values).
    gap_runs = _pad_and_merge_runs(gap_runs, gap_pad, n)
    for s, e in gap_runs:
        base_valid[s:e] = False

    def detect(signal, valid):
        """Peaks of ``signal`` lying within ``valid`` regions (searched on full array)."""
        vsig = signal[valid]
        if vsig.size == 0:
            return []
        vmax = float(np.max(vsig))
        if vmax <= 0:
            return []
        peaks, _ = find_peaks(signal, prominence=prominence_factor * vmax)
        return [int(p) for p in peaks if valid[p]]

    def valid_for(peaks):
        v = base_valid.copy()
        for p in peaks:
            v[max(int(p) - exclusion_window, 0):min(int(p) + exclusion_window + 1, n)] = False
        return v

    # --- Pre-fit peak-search iterations (on the smoothed profile) ---
    # One pass is always performed; `peak_iterations` is the number of ADDITIONAL
    # passes (mirrors how `iterations` counts additional refits after the initial fit).
    # Each pass excludes the peaks found so far, which lowers the running maximum so
    # progressively weaker peaks get caught. Gated by prominence_factor.
    detected = set()
    for _ in range(int(peak_iterations) + 1):
        valid = valid_for(detected)
        new = [p for p in detect(smoothed, valid) if p not in detected]
        if not new:
            break
        detected.update(new)

    # Initial fit with the pre-detected peaks excluded.
    (valid, real_tth, real_I, pseudo_tth, pseudo_I,
     sample_tth, sample_I) = _select_background_samples(
        twotheta, profile, base_valid, detected, exclusion_window, gap_runs)
    background = _fit_polynomial(twotheta, sample_tth, sample_I, poly_degree)

    # Residual refinement: find peaks the primary pass missed, exclude them, refit.
    # Missed peaks must clear a NOISE floor (>= 5 robust sigma above the residual
    # median), otherwise noise wiggles would be mistaken for peaks and runaway.
    for _ in range(int(iterations)):
        residual = profile - background
        # Only search where we currently treat the signal as background.
        residual_search = np.where(valid, residual, 0.0)
        residual_search[residual_search < 0] = 0.0
        if smoothing_sigma > 0:
            residual_search = gaussian_filter1d(residual_search, sigma=smoothing_sigma)
        valid_residual = residual_search[valid]
        if valid_residual.size == 0:
            break
        median = float(np.median(valid_residual))
        mad = float(np.median(np.abs(valid_residual - median)))
        sigma = 1.4826 * mad if mad > 0 else float(valid_residual.std())
        residual_max = float(np.max(valid_residual))
        if sigma <= 0 or residual_max <= 0:
            break
        # A missed peak must clear BOTH a noise floor (>= 5 robust sigma, prevents
        # noise runaway) AND `prominence_factor` of the largest remaining residual
        # feature (so raising prominence_factor above 1 disables detection, matching
        # the primary pass).
        prominence_threshold = max(prominence_factor * residual_max, 5.0 * sigma)
        peaks, _ = find_peaks(residual_search, prominence=prominence_threshold)
        # Reject candidates sitting next to an already-excluded peak region: those are
        # usually artifacts from subtracting a background that doesn't perfectly match
        # the peak shape, not genuine missed peaks. A candidate is discarded if it lies
        # within one exclusion window of the edge of an existing excluded peak (i.e.
        # within 2*exclusion_window of a detected peak centre).
        detected_arr = np.array(sorted(detected)) if detected else None
        guard = 2 * exclusion_window
        new = []
        for p in peaks:
            p = int(p)
            if not valid[p] or p in detected:
                continue
            if detected_arr is not None and detected_arr.size and \
                    np.min(np.abs(detected_arr - p)) < guard:
                continue
            new.append(p)
        if not new:
            break
        detected.update(new)
        (valid, real_tth, real_I, pseudo_tth, pseudo_I,
         sample_tth, sample_I) = _select_background_samples(
            twotheta, profile, base_valid, detected, exclusion_window, gap_runs)
        background = _fit_polynomial(twotheta, sample_tth, sample_I, poly_degree)

    if return_detail:
        peak_idx = (np.array(sorted(detected), dtype=int)
                    if detected else np.array([], dtype=int))
        return {
            "background": background,
            "real_tth": real_tth,
            "real_I": real_I,
            "pseudo_tth": pseudo_tth,
            "pseudo_I": pseudo_I,
            "valid_mask": valid,
            "peak_tth": twotheta[peak_idx],
            "peak_I": profile[peak_idx],
            "exclusion_window": int(exclusion_window),
        }
    return background


def compute_cake_background(
    cake: CakeData,
    *,
    n_bins: int = None,
    negative_clip: float = -10.0,
    **fit_kwargs,
) -> CakeBackground:
    """Fit and subtract a per-azimuth-bin polynomial background over the cake.

    The azimuth rows are grouped into ``n_bins`` equal bins. For each bin the raw
    rows are averaged into a single binned profile, the background is fit with
    :func:`fit_bin_background` (primary peak pass + residual refinement), and that
    single bin background is applied to *every* (finer-resolution) row in the bin and
    subtracted. So the user's azimuth binning directly controls how the background is
    estimated.

    Parameters
    ----------
    cake : CakeData
    n_bins : int, optional
        Number of azimuth bins. If None or >= the number of rows, each row is its
        own bin (finest resolution).
    negative_clip : float
        After subtraction, values below this are set to 0 (removes large negative
        excursions from data gaps).
    **fit_kwargs
        Forwarded to :func:`fit_bin_background` (poly_degree, smoothing_sigma,
        prominence_factor, peak_iterations, iterations, exclusion_window, gap_fill,
        gap_min_width, gap_pad).

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
        background[rows] = fit_bin_background(twotheta, profile, **fit_kwargs)

    subtracted = intensity - background
    subtracted[subtracted < negative_clip] = 0.0
    return CakeBackground(background=background, subtracted=subtracted)


def background_from_grid(cake: CakeData, background_grid, *,
                         negative_clip: float = -10.0) -> CakeBackground:
    """Build a :class:`CakeBackground` from an externally-supplied background grid.

    Used when the user loads a pre-made background instead of fitting one. The grid
    must match the cake's intensity shape.
    """
    intensity = np.asarray(cake.intensity, dtype=float)
    background = np.asarray(background_grid, dtype=float)
    subtracted = intensity - background
    subtracted[subtracted < negative_clip] = 0.0
    return CakeBackground(background=background, subtracted=subtracted)


def cake_grid_to_txt_bytes(twotheta, azimuth, grid) -> bytes:
    """Serialise a 2D grid back to the Dioptas cake .txt matrix layout.

    Round-trips with :func:`load_cake_data`: row 0 is ``[corner, 2th...]`` (the corner
    is a placeholder that load drops); each subsequent row is ``[azimuth, intensity...]``.
    """
    twotheta = np.asarray(twotheta, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n_az, n_tth = grid.shape
    matrix = np.zeros((n_az + 1, n_tth + 1), dtype=float)
    matrix[0, 1:] = twotheta          # matrix[0, 0] stays 0 (corner placeholder)
    matrix[1:, 0] = azimuth
    matrix[1:, 1:] = grid
    buffer = io.StringIO()
    np.savetxt(buffer, matrix, fmt="%.6g", delimiter=" ")
    return buffer.getvalue().encode("utf-8")


def grid_to_tiff_bytes(grid) -> bytes:
    """Serialise a 2D grid to a 32-bit float TIFF (preserves intensity values)."""
    arr = np.asarray(grid, dtype=np.float32)
    buffer = io.BytesIO()
    Image.fromarray(arr, mode="F").save(buffer, format="TIFF")
    return buffer.getvalue()


def load_grid_file(file, filename: str = "") -> np.ndarray:
    """Load a 2D grid from a cake ``.txt`` (Dioptas layout) or a ``.tiff`` image.

    For ``.txt`` the intensity grid is parsed via :func:`load_cake_data`; for a TIFF the
    pixel array is read directly (no embedded axes).
    """
    name = (filename or getattr(file, "name", "") or "").lower()
    if name.endswith((".tif", ".tiff")):
        return np.asarray(Image.open(file), dtype=float)
    return load_cake_data(file, filename).intensity


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
    cake: CakeData, grid, title: str, percentile: float = 99.5, highlight_band=None
) -> go.Figure:
    """Plot an arbitrary 2D grid (e.g. background or subtracted) on the cake axes.

    ``highlight_band`` (lo, hi) draws a translucent horizontal band across the plot at
    that azimuth range — used to show the currently-selected azimuth bin.
    """
    fig = _build_heatmap(cake.twotheta, cake.azimuth, grid, title, percentile)
    if highlight_band is not None:
        lo, hi = float(highlight_band[0]), float(highlight_band[1])
        fig.add_hrect(y0=lo, y1=hi, line_width=1, line_color="#00e5ff",
                      fillcolor="#00e5ff", opacity=0.3, layer="above")
    return fig


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
    detail = fit_bin_background(twotheta, profile, return_detail=True, **sample_kwargs)

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


# ===================================================================================
# 2D Refinement Tools — experimental peak extraction from a (background-subtracted) cake
# ===================================================================================

# Initial guess for the Pseudo-Voigt Gaussian<->Lorentzian fraction (gl), also the
# starting value the fit refines from. 0 = pure Gaussian, 1 = pure Lorentzian.
PSEUDOVOIGT_DEFAULT_GL = 0.5


def _gaussian(x, amp, center, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _pseudo_voigt(x, amp, center, sigma, gl):
    """FITYK-style Pseudo-Voigt: a ``gl``-weighted sum of a Gaussian and a Lorentzian.

    ``gl`` is the Gaussian->Lorentzian fraction (0 = pure Gaussian, 1 = pure Lorentzian).
    In this parametrisation the FWHM is ``2*sigma`` for any ``gl`` (both components share
    half-max at ``|x - center| = sigma``).
    """
    z = (x - center) / sigma
    gauss = np.exp(-np.log(2.0) * z * z)
    lorentz = 1.0 / (1.0 + z * z)
    return amp * ((1.0 - gl) * gauss + gl * lorentz)


def _fit_peak(x, y, peak_idx, peak_shape, fit_window):
    """Fit a single peak with the chosen shape in a window.

    Returns ``(center, amp, fwhm, gl)``. ``gl`` is the fitted Gaussian->Lorentzian
    fraction for a Pseudo-Voigt, or ``nan`` for a pure Gaussian. Falls back to the raw
    find_peaks position/height if the fit fails or the fitted centre leaves the window.
    """
    n = x.size
    lo = max(0, peak_idx - fit_window)
    hi = min(n, peak_idx + fit_window + 1)
    xw, yw = x[lo:hi], y[lo:hi]
    x0 = float(x[peak_idx])
    amp0 = float(max(y[peak_idx], 1e-9))
    sigma0 = max((float(x[hi - 1]) - float(x[lo])) / 6.0, 1e-3)
    try:
        if peak_shape == "Gaussian":
            popt, _ = curve_fit(_gaussian, xw, yw, p0=[amp0, x0, sigma0], maxfev=2000)
            amp, center, sigma = popt
            fwhm = 2.35482 * abs(sigma)     # Gaussian std -> FWHM
            gl = float("nan")
        else:
            popt, _ = curve_fit(
                _pseudo_voigt, xw, yw, p0=[amp0, x0, sigma0, PSEUDOVOIGT_DEFAULT_GL],
                bounds=([0.0, float(x[lo]), 1e-4, 0.0],
                        [np.inf, float(x[hi - 1]), np.inf, 1.0]), maxfev=3000)
            amp, center, sigma, gl = popt
            fwhm = 2.0 * abs(sigma)         # FITYK Pseudo-Voigt: FWHM = 2*sigma for any gl
            gl = float(gl)
        if not (x[lo] <= center <= x[hi - 1]):
            raise ValueError("fitted centre outside window")
        return float(center), float(amp), float(fwhm), gl
    except Exception:
        return x0, amp0, float("nan"), float("nan")


def seed_group_centres(cake: CakeData, grid, *, tth_min, tth_max, n_groups,
                       prominence=0.05, min_distance=1):
    """Seed group (hkl) ring centres from the azimuth max-projection lineout.

    Taking the max over azimuth at each 2th means even azimuthally-narrow (arc/spotty)
    rings still produce a peak, so every expected reflection can seed a group. Returns
    ``(seeds, strengths)``: up to ``n_groups`` seed 2th positions (strongest by
    prominence) and their projection heights, both sorted ascending in 2th.
    ``min_distance`` is the minimum separation (in 2th samples) between seeds, so two
    peaks of one ring are not seeded as separate groups.
    """
    twotheta = np.asarray(cake.twotheta, dtype=float)
    grid = np.asarray(grid, dtype=float)
    mask = (twotheta >= tth_min) & (twotheta <= tth_max)
    tth_win = twotheta[mask]
    empty = (np.array([], dtype=float), np.array([], dtype=float))
    if tth_win.size == 0:
        return empty
    proj = np.nanmax(grid[:, mask], axis=0)
    if proj.size == 0 or np.nanmax(proj) <= 0:
        return empty
    peaks, props = find_peaks(proj, prominence=prominence * float(np.nanmax(proj)),
                              distance=max(1, int(min_distance)))
    if peaks.size == 0:
        return empty
    keep = peaks[np.argsort(props["prominences"])[::-1][:int(n_groups)]]
    order = np.argsort(tth_win[keep])
    return tth_win[keep][order], proj[keep][order]


def default_max_shift(seeds) -> float:
    """Auto ring-track tolerance: ~0.4 x the minimum seed spacing (fallback 0.2 deg)."""
    seeds = np.asarray(seeds, dtype=float)
    if seeds.size < 2:
        return 0.2
    return float(0.4 * np.min(np.diff(np.sort(seeds))))


def extract_and_group_peaks(
    cake: CakeData,
    grid,
    *,
    tth_min: float,
    tth_max: float,
    n_bins: int,
    max_peaks: int,
    peak_shape: str = "PseudoVoigt",
    seed_prominence: float = 0.03,
    min_seed_distance: int = 1,
    detect_sigma: float = 5.0,
    fit_window: int = 15,
    max_shift=None,
):
    """Seed-guided per-ring peak extraction with azimuth tracking.

    1. Seed up to ``max_peaks`` ring centres from the azimuth max-projection (so weak/arc
       rings are still found regardless of a much stronger neighbour).
    2. Walk azimuth bins in order, keeping a running centre per ring. In each bin, for each
       ring, search a +/- ``max_shift`` window around its running centre for the local max;
       if it clears a noise floor (``detect_sigma`` x the robust sigma of the window) it is
       fitted (Gaussian/Pseudo-Voigt), assigned to that ring's group, and the running
       centre updated -- coupling later bins to earlier assignments. Rings with no
       qualifying peak in a bin keep their centre (gap) and can resume later.

    A noise-floor threshold (not one relative to a ring's own max) means a ring is detected
    wherever it rises clearly above background noise, so a strong hot-spot on one ring does
    not suppress detection of that ring elsewhere or of weaker rings. Yields one approximate
    2th per (bin, ring). Returns ``(peaks_df, seeds)`` with peaks_df columns
    ``bin, azimuth, 2th, intensity, fwhm, group``.
    """
    twotheta = np.asarray(cake.twotheta, dtype=float)
    grid = np.asarray(grid, dtype=float)
    seeds, strengths = seed_group_centres(
        cake, grid, tth_min=tth_min, tth_max=tth_max, n_groups=max_peaks,
        prominence=seed_prominence, min_distance=min_seed_distance)
    # Report the fitted Gaussian<->Lorentzian fraction only for the Pseudo-Voigt shape.
    want_gl = (peak_shape != "Gaussian")
    cols = ["bin", "azimuth", "2th", "intensity", "fwhm"]
    if want_gl:
        cols.append("gl")
    cols.append("group")
    if seeds.size == 0:
        return pd.DataFrame(columns=cols), seeds
    if max_shift is None:
        max_shift = default_max_shift(seeds)

    _edges, bin_index, _bw = assign_azimuth_bins(cake.azimuth, int(n_bins))
    mask = (twotheta >= tth_min) & (twotheta <= tth_max)
    tth_win = twotheta[mask]
    az = np.asarray(cake.azimuth, dtype=float)
    centres = seeds.astype(float).copy()
    # Absolute detection floor from the robust noise level of the (background-subtracted)
    # window: a ring is recorded in a bin only where its peak clears detect_sigma x sigma.
    win_vals = grid[:, mask]
    _med = float(np.median(win_vals))
    _mad = float(np.median(np.abs(win_vals - _med)))
    _sigma = 1.4826 * _mad if _mad > 0 else float(np.std(win_vals))
    threshold = _med + detect_sigma * _sigma

    rows = []
    for b in range(int(n_bins)):
        bin_rows = np.where(bin_index == b)[0]
        if bin_rows.size == 0:
            continue
        az_center = float(az[bin_rows].mean())
        lineout = grid[bin_rows][:, mask].mean(axis=0)
        if lineout.size == 0:
            continue
        for g in range(seeds.size):
            win = np.where(np.abs(tth_win - centres[g]) <= max_shift)[0]
            if win.size == 0:
                continue
            k = int(win[int(np.argmax(lineout[win]))])
            if lineout[k] <= 0 or lineout[k] < threshold:
                continue
            center, amp, fwhm, gl = _fit_peak(tth_win, lineout, k, peak_shape, fit_window)
            if abs(center - centres[g]) > max_shift:   # fit escaped the window
                center = float(tth_win[k])
            row = {"bin": int(b), "azimuth": az_center, "2th": center,
                   "intensity": amp, "fwhm": fwhm, "group": int(g)}
            if want_gl:
                row["gl"] = gl
            rows.append(row)
            centres[g] = center
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        # Order the table by hkl group, then azimuth, then 2th.
        df = df.sort_values(["group", "azimuth", "2th"]).reset_index(drop=True)
    return df, seeds


# Distinct colours for group overlays (repeat if more groups than colours).
def summarise_groups(peaks_df) -> pd.DataFrame:
    """Per-group summary to help assign hkl reflections.

    Returns one row per group with the number of points, the mean 2th (to identify the
    ring), and the azimuth coverage, sorted by mean 2th (ascending).
    """
    cols = ["group", "points", "mean_2th", "az_min", "az_max"]
    if peaks_df is None or peaks_df.empty or "group" not in peaks_df.columns:
        return pd.DataFrame(columns=cols)
    g = peaks_df.groupby("group")
    out = pd.DataFrame({
        "group": [int(k) for k in g.groups.keys()],
        "points": g.size().to_numpy(),
        "mean_2th": g["2th"].mean().to_numpy(),
        "az_min": g["azimuth"].min().to_numpy(),
        "az_max": g["azimuth"].max().to_numpy(),
    }).sort_values("mean_2th").reset_index(drop=True)
    return out


_GROUP_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
                 "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324", "#800000"]


def plot_extracted_peaks(cake: CakeData, grid, peaks_df, percentile: float = 99.5,
                         group_labels=None) -> go.Figure:
    """Subtracted-cake heatmap with extracted peaks scattered on top, coloured by group.

    ``group_labels`` (optional dict group->label) puts the assigned hkl label in the
    legend, e.g. ``group 0 · 111``.
    """
    group_labels = group_labels or {}
    fig = _build_heatmap(cake.twotheta, cake.azimuth, grid, "Extracted peaks", percentile)
    if peaks_df is not None and not peaks_df.empty and "group" in peaks_df.columns:
        for i, g in enumerate(sorted(peaks_df["group"].unique())):
            sub = peaks_df[peaks_df["group"] == g]
            color = "#9e9e9e" if g == -1 else _GROUP_COLORS[i % len(_GROUP_COLORS)]
            name = "unassigned" if g == -1 else f"group {int(g)}"
            label = str(group_labels.get(int(g), "")).strip()
            if label:
                name = f"{name} · {label}"
            fig.add_trace(go.Scatter(
                x=sub["2th"], y=sub["azimuth"], mode="markers", name=name,
                marker=dict(color=color, size=5, line=dict(width=0.5, color="black"))))
    return fig


def _group_color_map(peaks_df) -> dict:
    """Map each group to a stable colour (matching :func:`plot_extracted_peaks`)."""
    cmap = {}
    if peaks_df is not None and not peaks_df.empty and "group" in peaks_df.columns:
        for i, g in enumerate(sorted(peaks_df["group"].unique())):
            cmap[int(g)] = "#9e9e9e" if g == -1 else _GROUP_COLORS[i % len(_GROUP_COLORS)]
    return cmap


def plot_bin_peak_fits(cake: CakeData, grid, peaks_df, *, bin_index, n_bins,
                       tth_min, tth_max, group_labels=None) -> go.Figure:
    """1D lineout of one azimuth bin with the extracted peak fits overlaid.

    The bin's averaged (background-subtracted) 2θ profile is drawn, and each peak found
    in that bin is reconstructed from its stored ``(2th, intensity, fwhm, gl)`` — a FITYK
    Pseudo-Voigt when a ``gl`` column is present (FWHM = 2·sigma), otherwise a Gaussian
    (FWHM = 2.35482·sigma) — so the curves mirror the results table. The bin lineout, each
    fitted peak (coloured by group, matching the overlay plot), their sum (composite), and
    any fallback (unfitted) peak centres are separate legend-toggleable traces.
    """
    group_labels = group_labels or {}
    twotheta = np.asarray(cake.twotheta, dtype=float)
    grid = np.asarray(grid, dtype=float)
    az = np.asarray(cake.azimuth, dtype=float)
    mask = (twotheta >= tth_min) & (twotheta <= tth_max)
    tth_win = twotheta[mask]

    _edges, bidx, _bw = assign_azimuth_bins(az, int(n_bins))
    rows_in_bin = np.where(bidx == int(bin_index))[0]
    if rows_in_bin.size and tth_win.size:
        lineout = grid[rows_in_bin][:, mask].mean(axis=0)
        az_lo, az_hi = float(az[rows_in_bin].min()), float(az[rows_in_bin].max())
        title = (f"Bin {int(bin_index)} peak fits — azimuth {az_lo:.1f}° to {az_hi:.1f}° "
                 f"(mean of {rows_in_bin.size} rows)")
    else:
        lineout = np.zeros(tth_win.size)
        title = f"Bin {int(bin_index)} — no data"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tth_win, y=lineout, name="Bin lineout", mode="lines",
                             line=dict(color="#444444", width=1.5)))

    color_map = _group_color_map(peaks_df)
    sub = (peaks_df[peaks_df["bin"] == int(bin_index)]
           if peaks_df is not None and not peaks_df.empty and "bin" in peaks_df.columns
           else None)
    is_pv = sub is not None and "gl" in sub.columns
    composite = np.zeros(tth_win.size)
    n_fits = 0
    if sub is not None and tth_win.size:
        for _, r in sub.sort_values("2th").iterrows():
            g = int(r["group"])
            center, amp, fwhm = float(r["2th"]), float(r["intensity"]), float(r["fwhm"])
            color = color_map.get(g, _GROUP_COLORS[0])
            label = str(group_labels.get(g, "")).strip()
            gname = "unassigned" if g == -1 else f"group {g}"
            if label:
                gname = f"{gname} · {label}"
            gl = float(r["gl"]) if (is_pv and "gl" in r) else float("nan")
            if np.isfinite(fwhm) and fwhm > 0 and np.isfinite(amp):
                if is_pv and np.isfinite(gl):
                    curve = _pseudo_voigt(tth_win, amp, center, fwhm / 2.0, gl)
                else:
                    curve = _gaussian(tth_win, amp, center, fwhm / 2.35482)
                composite = composite + curve
                n_fits += 1
                fig.add_trace(go.Scatter(
                    x=tth_win, y=curve, name=f"{gname} @ {center:.3f}°", mode="lines",
                    line=dict(color=color, width=1.5, dash="dot")))
            else:
                # Fit failed / fell back to the raw position: mark the centre instead.
                fig.add_trace(go.Scatter(
                    x=[center], y=[amp if np.isfinite(amp) else 0.0],
                    name=f"{gname} @ {center:.3f}° (raw)", mode="markers",
                    marker=dict(color=color, size=9, symbol="triangle-down",
                                line=dict(width=1, color="black"))))
    if n_fits > 1:
        fig.add_trace(go.Scatter(x=tth_win, y=composite, name="Composite fit",
                                 mode="lines", line=dict(color="black", width=1, dash="dash"),
                                 visible="legendonly"))

    fig.update_layout(
        title=title, xaxis_title="2th (degrees)", yaxis_title="Intensity",
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


# =========================================================================
# REFINEMENT STAGES
#
# Three stages fit progressively more of the model, each seeded from the last:
#   1  peak POSITIONS   2th vs azimuth      -> lattice, stress, chi
#   2  peak INTENSITIES integrated area     -> preferred orientation
#   3  the IMAGE itself blocked 2th/azimuth -> all of the above plus peak width
#
# Shared machinery, so a fix lands in one place for all three:
#   _run_refinement   the lmfit/scipy driver, at-limit reporting and fit statistics
#   param_bounds      every parameter's default limits
#   _plot_panel_grid  the per-hkl data-vs-model panel grids
#
# spinel_core.compute_strain is INJECTED as ``strain_fn`` rather than imported: it pulls
# in pyFAI and lmfit and itself imports this module, so importing it here would be
# circular and would stop cake_processing loading in a bare environment.
# =========================================================================

# --- Stage 1: simulated "Mean two_th @ delta" vs the extracted peak positions -----------

REFINEMENT_HELP_MD = """\
### Stage 1 refinement — peak positions

For every hkl this compares the **experimental** mean 2θ at each azimuth bin (loaded from
a peak-fit CSV) with the **simulated** `Mean two_th @ delta` from `compute_strain`, using
the model set up on the **Simulation** tab (symmetry, elastic constants, PO, wavelength, χ).

- The simulated curve is evaluated on its own azimuth (δ) grid and **interpolated**
  (periodically, 360°) onto the experimental azimuths, then residuals `data − sim` are
  minimised by least squares over the parameters you enable.
- **Parameters** (each toggleable): lattice lengths `a` (and `b`, `c` where the symmetry
  needs them), the six stress components `σ11…σ23`, and `χ`. A good first pass is
  `a`, `σ11`, `σ33` only, with the other stresses fixed at 0. The differential stress
  **t = σ33 − σ11** is reported from the refined values.
- Only hkls present in **both** the CSV (with an assigned label) and the Simulation-tab
  hkl list can be used. The **Refine** and **Plot** tick-lists are independent, so you can
  fit one subset while viewing another (excluded-but-plotted rings are greyed).
- The optimiser, its limits and the fit report are described under
  *Refinement engine & limits*.
"""

SIGMA_NAMES = ["sigma_11", "sigma_22", "sigma_33", "sigma_12", "sigma_13", "sigma_23"]


def lattice_param_names(symmetry) -> list:
    """Independent (refineable) lattice-length parameters for a symmetry.

    The dependent lengths are coupled to these by :func:`apply_symmetry_constraints`
    (e.g. cubic b, c inherit a). Angles are fixed in Stage 1.
    """
    s = (symmetry or "").lower()
    if s.startswith("cubic"):
        return ["a_val"]
    if s.startswith(("hex", "tetra", "trig")):
        return ["a_val", "c_val"]
    return ["a_val", "b_val", "c_val"]


def apply_symmetry_constraints(symmetry, lattice_params) -> dict:
    """Couple the dependent lattice lengths to ``a`` for the given crystal symmetry.

    - cubic: ``b = c = a``
    - hexagonal / tetragonal / trigonal: ``b = a`` (c independent)
    - orthorhombic (and anything else): ``a``, ``b``, ``c`` left independent

    Angles are untouched. Applied after overlaying refined values so a refined ``a``
    always propagates to its symmetry-locked partners.
    """
    lp = dict(lattice_params)
    s = (symmetry or "").lower()
    a = lp.get("a_val")
    if s.startswith("cubic"):
        lp["b_val"] = a
        lp["c_val"] = a
    elif s.startswith(("hex", "tetra", "trig")):
        lp["b_val"] = a
    return lp


def hkl_label(hkl) -> str:
    """Canonical string label for an (h, k, l) tuple, matching compute_strain."""
    h, k, l = hkl
    return f"{int(h)}{int(k)}{int(l)}"


def normalize_hkl_label(value) -> str:
    """Normalise an assigned hkl label so it matches the simulated ``f'{h}{k}{l}'`` form.

    Handles CSV round-trips where a numeric hkl column is parsed as a float (``111`` ->
    ``111.0``) and strips surrounding whitespace; blanks/NaN become ``""``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("", "nan"):
        return ""
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    return s


def load_labelled_peaks_csv(file, filename=None) -> pd.DataFrame:
    """Load a peak-fit CSV (as exported by the Peak Extraction tab).

    Requires ``azimuth``, ``2th`` and ``hkl`` columns; rows with a blank/NaN hkl are
    dropped (they cannot be matched to a simulated reflection).
    """
    df = pd.read_csv(file)
    needed = {"azimuth", "2th", "hkl"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            "CSV missing column(s): " + ", ".join(sorted(missing)) +
            ". Expected the peak-fit export (azimuth, 2th, hkl, ...).")
    df = df.copy()
    df["hkl"] = df["hkl"].map(normalize_hkl_label)
    df = df[df["hkl"] != ""]
    return df.reset_index(drop=True)


def experimental_hkl_curves(peaks_df, azimuth_col="azimuth", tth_col="2th",
                            hkl_col="hkl") -> dict:
    """Group labelled peaks by hkl and average 2θ per azimuth bin.

    Returns ``{hkl_label: DataFrame[azimuth, mean_2th, n]}`` sorted by azimuth. Points
    sharing an (hkl, azimuth) bin are averaged.
    """
    out = {}
    if peaks_df is None or peaks_df.empty or hkl_col not in peaks_df.columns:
        return out
    for hkl, sub in peaks_df.groupby(hkl_col):
        g = (sub.groupby(azimuth_col)[tth_col].agg(["mean", "size"]).reset_index()
             .rename(columns={azimuth_col: "azimuth", "mean": "mean_2th", "size": "n"})
             .sort_values("azimuth").reset_index(drop=True))
        out[str(hkl)] = g
    return out


def interp_periodic(x_known, y_known, x_query, period=360.0):
    """Linear interpolation of a periodic curve (azimuth in degrees, period 360).

    ``x_known`` need not be sorted; the data is tiled one period each side so queries
    between the last and first sample interpolate across the wrap point.
    """
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    order = np.argsort(x_known)
    xs, ys = x_known[order], y_known[order]
    xe = np.concatenate([xs - period, xs, xs + period])
    ye = np.concatenate([ys, ys, ys])
    return np.interp(np.asarray(x_query, dtype=float), xe, ye)


def _sim_curve_for_hkl(strain_fn, hkl, intensity, symmetry, lattice_params, wavelength,
                       cijs, sigma_params, chi, po_model=None, coarse=False):
    """Simulated (delta, mean_2th) curve for one hkl via the injected compute_strain."""
    phi_values = np.radians(np.arange(0, 360, 5))
    psi_values = 1 if coarse else 0    # compute_strain: 0 -> 2 deg deltas, nonzero -> 12 deg
    label, df, _psi, _s = strain_fn(
        hkl, intensity, symmetry, lattice_params, wavelength, cijs, sigma_params,
        chi, phi_values, psi_values, po_model=po_model)
    d = (df[["delta (degrees)", "Mean two_th @ delta"]].dropna()
         .drop_duplicates("delta (degrees)").sort_values("delta (degrees)"))
    return d["delta (degrees)"].to_numpy(), d["Mean two_th @ delta"].to_numpy(), label


def simulate_all_curves(strain_fn, sim_context, lattice_params, sigma_params, chi,
                        hkl_labels=None, coarse=False) -> dict:
    """Compute ``{hkl_label: (delta, mean_2th)}`` for the requested hkls."""
    out = {}
    for hkl, inten in zip(sim_context["selected_hkls"], sim_context["intensities"]):
        label = hkl_label(hkl)
        if hkl_labels is not None and label not in hkl_labels:
            continue
        delta, sim2th, _ = _sim_curve_for_hkl(
            strain_fn, hkl, inten, sim_context["symmetry"], lattice_params,
            sim_context["wavelength"], sim_context["cijs"], sigma_params, chi,
            po_model=sim_context.get("po_model"), coarse=coarse)
        out[label] = (delta, sim2th)
    return out


def _assemble_params(sim_context, values):
    """Overlay refineable ``values`` onto the sim_context base lattice/sigma/chi."""
    lattice_params = dict(sim_context["lattice_params"])
    for key in ("a_val", "b_val", "c_val", "alpha", "beta", "gamma"):
        if key in values:
            lattice_params[key] = float(values[key])
    # Enforce symmetry coupling so a refined `a` propagates to b/c as required.
    lattice_params = apply_symmetry_constraints(sim_context.get("symmetry"), lattice_params)
    sigma_params = {n: float(values.get(n, sim_context["sigma_params"].get(n, 0.0)))
                    for n in SIGMA_NAMES}
    chi = float(values.get("chi", sim_context["chi"]))
    return lattice_params, sigma_params, chi


#: Lattice lengths are bounded RELATIVE to the starting value (+/- this fraction), so a
#: poor initial guess limits how far the refinement can travel -- widen it, or set explicit
#: limits in the UI, when starting far from the answer. Everything else is absolute.
LATTICE_BOUND_FRAC = 0.25
SIGMA_BOUNDS = (-25.0, 25.0)          # GPa
CHI_BOUNDS = (-90.0, 90.0)            # degrees
FWHM_BOUNDS = (1e-3, 5.0)             # degrees
#: March-Dollase limits (Stage 2 and 3). R = 1 is isotropic.
PO_PARAM_BOUNDS = {"R": (0.05, 5.0), "tau": (-180.0, 180.0),
                   "omega": (-180.0, 180.0), "baseline": (0.0, 1.0)}


def param_bounds(name, value, lattice_frac=LATTICE_BOUND_FRAC):
    """Default ``(min, max)`` refinement limits for any parameter, in any stage.

    One table for all three stages so a limit cannot drift between them. Lattice lengths
    get a window around the START value; stress, chi, fwhm and the PO parameters use fixed
    physical ranges. Unknown names are unbounded.
    """
    if name in ("a_val", "b_val", "c_val"):
        if value:
            return ((1.0 - lattice_frac) * value, (1.0 + lattice_frac) * value)
        return (0.0, np.inf)
    if name.startswith("sigma_"):
        return SIGMA_BOUNDS
    if name == "chi":
        return CHI_BOUNDS
    if name == "fwhm":
        return FWHM_BOUNDS
    return PO_PARAM_BOUNDS.get(name, (-np.inf, np.inf))



def evaluate_curves_and_residuals(strain_fn, sim_context, exp_curves, values,
                                  coarse=False) -> dict:
    """Simulate every included hkl at ``values`` and score it against the experiment.

    Returns ``sim_curves`` (for plotting), per-hkl RMSE, overall RMSE and the derived
    differential stress ``t = sigma_33 - sigma_11``.
    """
    lattice_params, sigma_params, chi = _assemble_params(sim_context, values)
    labels = set(exp_curves.keys())
    sims = simulate_all_curves(strain_fn, sim_context, lattice_params, sigma_params, chi,
                               hkl_labels=labels, coarse=coarse)
    per_hkl, all_res = {}, []
    for label, g in exp_curves.items():
        if label not in sims:
            per_hkl[label] = float("nan")
            continue
        delta, sim2th = sims[label]
        resid = g["mean_2th"].to_numpy() - interp_periodic(delta, sim2th,
                                                           g["azimuth"].to_numpy())
        per_hkl[label] = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
        all_res.append(resid)
    rmse = float(np.sqrt(np.mean(np.concatenate(all_res) ** 2))) if all_res else float("nan")
    return {"sim_curves": sims, "per_hkl": per_hkl, "rmse": rmse,
            "t": float(sigma_params["sigma_33"] - sigma_params["sigma_11"]),
            "lattice_params": lattice_params, "sigma_params": sigma_params, "chi": chi}


#: Optimiser methods offered for Stage 1 (lmfit names).
REFINEMENT_METHODS = ["leastsq", "least_squares", "nelder"]

ENGINE_HELP_MD = """\
### Refinement engine & settings

**Engine — lmfit.** Stage 1 uses [lmfit](https://lmfit.github.io/lmfit-py/)'s `minimize`,
the same library the 1D pattern refinement uses, so the full **fit report** (parameter
values with ±1σ, initial values, bounds, correlations, χ², reduced χ², AIC/BIC, function
evaluations) is produced and shown below the results table.

**Method**
- `leastsq` — Levenberg–Marquardt (default). Fast, gives a covariance matrix, so
  uncertainties and correlations are reported. Best for a well-behaved starting point.
- `least_squares` — SciPy Trust Region Reflective. More robust when parameters sit near
  their limits, but sometimes reports no covariance.
- `nelder` — Nelder–Mead simplex. Derivative-free; useful when the residual is noisy or
  the start is poor, but it gives **no uncertainties** and is slower to converge.

**Parameter limits (this is a real constraint).** Every refined parameter is bounded:
- **Lattice a/b/c** — bounded to a window of **±(fraction) of the starting value**
  (default ±25 %). The window is centred on your *initial guess*, so if the guess is far
  from the truth the refinement **cannot travel beyond it** — e.g. starting at 4.0 Å with
  a ±25 % window can only reach 3.0–5.0 Å and will stop at the limit. If a refined value
  lands exactly on its min/max, widen the fraction (or set explicit limits) and re-run.
- **σ components** — ±25 GPa by default.
- **χ** — ±90°.
All limits can be overridden per parameter under *Limits*, and `Max evaluations`
(`max_nfev`) caps the number of forward-model calls.

**Convergence.** `leastsq`/`least_squares` stop on `ftol`/`xtol`/`gtol` (~1e-8). The
report's message states which condition ended the fit; "at initial value" next to a
parameter in the report means it never moved (usually a flat gradient or a bound).
"""


def _run_refinement(init_values, refine_flags, residual_fn, *, limits_fn,
                    method="leastsq", max_nfev=200, n_points=0, derived=None) -> dict:
    """Least-squares driver shared by all three refinement stages.

    Each stage supplies only what makes it different:

    - ``residual_fn(values)`` -> the residual array, or ``(array, extras)`` when the stage
      computes something worth reporting alongside it (Stages 2 and 3 profile out a global
      scale this way, so it comes back without a second forward-model call).
    - ``limits_fn(name, value)`` -> that parameter's ``(min, max)``.
    - ``derived(values)`` -> optional cheap extra result fields (Stage 1's ``t``, the
      phi-sampling advice in Stages 2 and 3).

    Uses lmfit when installed, for the fit report, uncertainties and correlations; falls
    back to ``scipy.optimize.least_squares`` otherwise. Always returns the same shape of
    dict: ``values, errors, success, message, rmse, n_points, n_free, report, at_limit,
    method`` plus the fit statistics and whatever the stage contributed.
    """
    free = [n for n, on in refine_flags.items() if on and n in init_values]
    limits = {n: limits_fn(n, init_values[n]) for n in free}

    def resid_and_extras(values):
        out = residual_fn(values)
        if isinstance(out, tuple):
            return np.asarray(out[0], dtype=float), dict(out[1] or {})
        return np.asarray(out, dtype=float), {}

    def package(values, errors, success, message, resid, extras, report="",
                at_limit=None, stats=None):
        out = {"values": values, "errors": errors, "success": bool(success),
               "message": str(message),
               "rmse": float(np.sqrt(np.mean(np.asarray(resid) ** 2))),
               "n_points": int(n_points), "n_free": len(free),
               "report": report, "at_limit": at_limit or [], "method": method}
        out.update(extras)
        if derived is not None:
            out.update(derived(values))
        out.update(stats or {})
        return out

    if not free:
        resid, extras = resid_and_extras(dict(init_values))
        return package(dict(init_values), {}, True,
                       "No parameters selected to refine.", resid, extras)

    try:
        from lmfit import Parameters, minimize as lm_minimize, fit_report as lm_fit_report
    except ImportError:
        Parameters = None

    if Parameters is not None:
        lm_params = Parameters()
        for n in free:
            lo, hi = limits[n]
            lm_params.add(n, value=float(np.clip(init_values[n], lo, hi)),
                          min=lo, max=hi, vary=True)

        def lm_residual(p):
            values = dict(init_values)
            values.update({n: p[n].value for n in free})
            return resid_and_extras(values)[0]

        result = lm_minimize(lm_residual, lm_params, method=method, max_nfev=max_nfev)
        refined, errors, at_limit = dict(init_values), {}, []
        for n in free:
            par = result.params[n]
            refined[n] = float(par.value)
            errors[n] = float(par.stderr) if par.stderr is not None else float("nan")
            lo, hi = limits[n]
            if np.isfinite(lo) and np.isclose(par.value, lo, rtol=1e-6, atol=1e-9):
                at_limit.append(f"{n} (at min {lo:g})")
            elif np.isfinite(hi) and np.isclose(par.value, hi, rtol=1e-6, atol=1e-9):
                at_limit.append(f"{n} (at max {hi:g})")
        resid, extras = resid_and_extras(refined)
        stats = {k: float(getattr(result, k, np.nan))
                 for k in ("chisqr", "redchi", "aic", "bic")}
        stats["nfev"] = int(getattr(result, "nfev", 0))
        return package(refined, errors, getattr(result, "success", True),
                       getattr(result, "message", ""), resid, extras,
                       report=lm_fit_report(result), at_limit=at_limit, stats=stats)

    lo = np.array([limits[n][0] for n in free], dtype=float)
    hi = np.array([limits[n][1] for n in free], dtype=float)
    x0 = np.clip(np.array([init_values[n] for n in free], dtype=float), lo, hi)

    def residuals_vec(vec):
        values = dict(init_values)
        values.update({n: v for n, v in zip(free, vec)})
        return resid_and_extras(values)[0]

    result = least_squares(residuals_vec, x0, bounds=(lo, hi), max_nfev=max_nfev)
    refined = dict(init_values)
    refined.update({n: float(v) for n, v in zip(free, result.x)})
    resid, extras = resid_and_extras(refined)
    return package(refined, _stderr_from_result(result, free), result.success,
                   result.message, resid, extras,
                   report="lmfit not installed — used scipy.optimize.least_squares.")


def _stderr_from_result(result, names):
    """1-sigma parameter errors from a scipy least_squares Jacobian (fallback path)."""
    try:
        J = result.jac
        dof = max(1, J.shape[0] - J.shape[1])
        cov = np.linalg.inv(J.T @ J) * (2.0 * result.cost / dof)
        se = np.sqrt(np.abs(np.diag(cov)))
        return {n: float(s) for n, s in zip(names, se)}
    except Exception:
        return {n: float("nan") for n in names}


def run_stage1_refinement(strain_fn, sim_context, exp_curves, init_values, refine_flags,
                          coarse=True, max_nfev=200, bounds=None, method="leastsq",
                          lattice_frac=LATTICE_BOUND_FRAC) -> dict:
    """Refine peak POSITIONS (2th vs azimuth) across the included hkls.

    ``init_values`` holds a start for every refineable parameter (lattice a/b/c, the six
    sigma components, chi); ``refine_flags`` selects which vary. ``bounds`` overrides the
    per-parameter limits -- note the lattice window is relative to the START value, so a
    far-off initial guess restricts how far the fit can travel.

    Adds ``t`` (= sigma_33 - sigma_11) to the shared result dict; see
    :func:`_run_refinement`.
    """
    bounds = dict(bounds or {})
    labels = list(exp_curves.keys())
    az = {L: exp_curves[L]["azimuth"].to_numpy() for L in labels}
    y = {L: exp_curves[L]["mean_2th"].to_numpy() for L in labels}

    def residual(values):
        lattice_params, sigma_params, chi = _assemble_params(sim_context, values)
        sims = simulate_all_curves(strain_fn, sim_context, lattice_params, sigma_params,
                                   chi, hkl_labels=set(labels), coarse=coarse)
        res = [y[L] - interp_periodic(*sims[L], az[L]) for L in labels if L in sims]
        return np.concatenate(res) if res else np.zeros(1)

    return _run_refinement(
        init_values, refine_flags, residual,
        limits_fn=lambda n, v: bounds.get(n, param_bounds(n, v, lattice_frac)),
        method=method, max_nfev=max_nfev,
        n_points=int(sum(v.size for v in y.values())),
        derived=lambda v: {"t": float(v.get("sigma_33", 0.0) - v.get("sigma_11", 0.0))})


def _subplot_vspace(nrows, row_height, gap_px=120):
    """Vertical spacing fraction giving at least ``gap_px`` between subplot rows.

    Plotly's ``vertical_spacing`` is a fraction of the TOTAL figure height, so a fixed
    fraction shrinks in absolute terms as rows are added — subplot titles then collide
    with the axis labels of the row above. Convert a target pixel gap into the fraction
    (never below the old 0.12 default) and respect plotly's 1/(nrows-1) ceiling.
    """
    if nrows <= 1:
        return 0.0
    frac = max(0.12, gap_px / float(row_height * nrows))
    return float(min(frac, 0.9 / (nrows - 1)))


#: Panel colours shared by every data-vs-model grid. Greyed = excluded from the fit.
_DATA_COLOR, _MODEL_COLOR = "#e6194B", "#4363d8"
_OFF_DATA_COLOR, _OFF_MODEL_COLOR = "#c9c9c9", "#dddddd"


def _plot_panel_grid(panels, *, title, y_title, x_title="azimuth (°)", ncols=4,
                     included=None, row_height=300, data_mode="lines+markers",
                     model_dash="dash") -> go.Figure:
    """One panel per hkl, ``ncols`` wide, each comparing measured data to the model.

    ``panels`` maps label -> ``(x_data, y_data, x_model, y_model)``; either pair may be
    None. Both series are sorted by x, so callers need not. Labels missing from
    ``included`` are greyed, marking rings shown but not fitted. Shared by the Stage 1,
    2 and 3 grids so they stay visually consistent and there is one place to fix.
    """
    labels = list(panels.keys())
    if not labels:
        return go.Figure()
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(len(labels) / ncols))
    included = set(labels) if included is None else set(included)
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"hkl {L}" for L in labels],
                        horizontal_spacing=0.06,
                        vertical_spacing=_subplot_vspace(nrows, row_height))
    for i, L in enumerate(labels):
        r, c = i // ncols + 1, i % ncols + 1
        on = L in included
        xd, yd, xm, ym = panels[L]
        if xd is not None:
            col = _DATA_COLOR if on else _OFF_DATA_COLOR
            o = np.argsort(np.asarray(xd, dtype=float))
            fig.add_trace(go.Scatter(
                x=np.asarray(xd)[o], y=np.asarray(yd)[o], mode=data_mode,
                name=f"{L} data", showlegend=False,
                marker=dict(color=col, size=5), line=dict(color=col, width=1)),
                row=r, col=c)
        if xm is not None:
            o = np.argsort(np.asarray(xm, dtype=float))
            fig.add_trace(go.Scatter(
                x=np.asarray(xm)[o], y=np.asarray(ym)[o], mode="lines",
                name=f"{L} model", showlegend=False,
                line=dict(color=_MODEL_COLOR if on else _OFF_MODEL_COLOR, width=2,
                          dash=model_dash)), row=r, col=c)
    fig.update_xaxes(title_text=x_title, title_standoff=6)
    fig.update_yaxes(title_text=y_title, title_standoff=6)
    fig.update_layout(height=row_height * nrows, margin=dict(l=70, r=20, t=70, b=50),
                      title=title)
    return fig


def plot_refinement_grid(exp_curves, sim_curves, ncols=4, included=None,
                         row_height=300) -> go.Figure:
    """Stage 1: 2th vs azimuth per hkl, measured points against the simulated curve."""
    return _plot_panel_grid(
        {L: (g["azimuth"].to_numpy(), g["mean_2th"].to_numpy(),
             *(sim_curves[L] if L in sim_curves else (None, None)))
         for L, g in exp_curves.items()},
        title="2θ vs azimuth — data (points) vs model (dashed)",
        y_title="2θ (°)", ncols=ncols, included=included, row_height=row_height,
        data_mode="markers")


# --- Stage 2: preferred orientation from the azimuthal intensity variation -------------
#
# Measured intensity is the integrated peak AREA (from amp/fwhm/gl), so it is unbiased
# when texture broadens the arcs. The model calls PO.PO_Model directly and averages over
# phi -- the same quantity compute_strain builds into "Mean I @ delta", but reached
# without going through it, which is why PO parameters here need no session-state hook.
# A single global scale, solved in closed form each iteration, absorbs the arbitrary
# experimental units while leaving the relative hkl intensities to constrain the fit.

#: phi-integration sampling tiers as (max MD sharpness, n_phi). The March-Dollase peak
#: contrast goes as max(R, 1/R)**4.5 -- sharp for SMALL R (peak at eta=0, P(0)=R^-3) and
#: also for LARGE R (band at eta=90, P=R^1.5) -- so the requirement is symmetric in
#: max(R, 1/R). Measured error vs a 2880-point reference (worst hkl, % of the modulation):
#:   R=0.5/1.5 -> 36: 0.003%   R=0.3 -> 72: 0.005%   R=5 -> 144: 0.0007%
#:   R=0.1 -> 288: 0.055%      R=0.05 -> 576: 0.49% (extreme edge of the allowed range)
#: A fixed 36 points would leave 1.7% error at R=0.3 and 7.6% at R=5.
PHI_SAMPLING_TIERS = ((2.0, 36), (4.0, 72), (6.0, 144), (12.0, 288))
PHI_SAMPLING_MAX = 576


def adaptive_n_phi(R) -> int:
    """Number of phi integration points needed to resolve the PO surface for a given R.

    Keyed on the March-Dollase sharpness ``max(R, 1/R)`` (see :data:`PHI_SAMPLING_TIERS`),
    which is what sets how narrow the features being integrated over are.
    """
    try:
        R = float(R)
    except (TypeError, ValueError):
        return PHI_SAMPLING_TIERS[0][1]
    if not np.isfinite(R) or R <= 0:
        return PHI_SAMPLING_MAX
    sharpness = max(R, 1.0 / R)
    for s_max, n in PHI_SAMPLING_TIERS:
        if sharpness <= s_max:
            return n
    return PHI_SAMPLING_MAX

STAGE2_HELP_MD = """\
### Stage 2 refinement — preferred orientation from intensities

Stage 1 fixed *where* the rings sit (2θ vs azimuth); Stage 2 fits *how strong* they are
around each ring, which is what preferred orientation (texture) controls.

**Experimental intensity.** Each extracted peak already carries an amplitude, FWHM and
(for Pseudo-Voigt) the Gaussian↔Lorentzian fraction, so the **integrated area** is used
rather than the raw amplitude — the physically meaningful reflection intensity, and
unbiased when texture broadens the arcs:
`area = amp · σ · [(1−gl)·√(π/ln2) + gl·π]` for the FITYK Pseudo-Voigt (σ = FWHM/2), and
`area = amp · σ_G · √(2π)` for a Gaussian (σ_G = FWHM/2.35482).

**Simulated intensity.** For each hkl the March–Dollase model is evaluated over
(φ, δ) and averaged over φ, giving PO intensity vs azimuth, multiplied by that hkl's
structure intensity from the Simulation tab.

**Scaling.** Experimental intensities are on an arbitrary scale, so **one global scale
factor** is applied across all hkls, solved analytically each iteration
(`s = Σ I_obs·I_sim / Σ I_sim²`). Because it is *global*, the relative intensities
between reflections still constrain the fit, and it costs no fit parameter.

**Parameters.** `R` (March–Dollase strength; R = 1 is no texture), `tau`/`omega` (the
preferred-orientation axis direction, degrees) and `baseline` (isotropic fraction, 0–1).

⚠️ **Refine in stages — and do not start the axis from R = 1.** At exactly R = 1 the model
is isotropic, so `tau` and `omega` have *zero* gradient: the axis direction is
mathematically unidentifiable there and the optimiser will wander into a false minimum.
The reliable recipe is:

1. **R only** (from any R ≠ 1, e.g. 0.9) — sets the texture strength.
2. **R + tau + omega** — the axis now has a real gradient to follow.
3. **add baseline** — finally the isotropic offset.

Each stage starts from the previous stage's refined values (press *Run* again after
ticking the next parameter). Skipping to "everything at once" from an isotropic start is
the main way this refinement goes wrong.

⚠️ **Do not start a parameter on its limit.** `baseline` is bounded at 0, so starting it at
exactly 0 freezes it — begin at ~0.05 instead. Note `baseline` is also weakly determined
unless the data are clean: check its reported ±1σ before trusting the value.

**φ sampling.** The PO intensity at each azimuth is an average over the free rotation φ,
evaluated on a discrete grid. How fine that grid must be depends on how *peaked* the
March–Dollase function is — sharp both for small R (a spike at η = 0) and for large R (a
narrow band at η = 90°) — so the requirement scales with `max(R, 1/R)`. The sampling is
therefore chosen **automatically from R** (36 points near R = 1, up to 576 at the extremes);
a fixed 36 would leave ~1.7 % error at R = 0.3 and ~7.6 % at R = 5. Set *φ sampling* to a
non-zero value to override and check stability. During a fit the value is held fixed
(switching mid-refinement would corrupt the gradients); if the refined R needs finer
sampling you'll be told to re-run.
"""


def peak_area(amp, fwhm, gl=None):
    """Integrated area of a fitted peak from its amplitude and FWHM.

    Uses the FITYK Pseudo-Voigt convention (sigma = FWHM/2, area =
    ``amp*sigma*[(1-gl)*sqrt(pi/ln2) + gl*pi]``) when ``gl`` is given and finite,
    otherwise the Gaussian result (``amp*sigma_G*sqrt(2*pi)``, sigma_G = FWHM/2.35482).
    """
    amp = np.asarray(amp, dtype=float)
    fwhm = np.asarray(fwhm, dtype=float)
    gauss_area = amp * (fwhm / 2.35482) * np.sqrt(2.0 * np.pi)
    if gl is None:
        return gauss_area
    gl = np.asarray(gl, dtype=float)
    sigma = fwhm / 2.0
    pv_area = amp * sigma * ((1.0 - gl) * np.sqrt(np.pi / np.log(2.0)) + gl * np.pi)
    return np.where(np.isfinite(gl), pv_area, gauss_area)


def experimental_intensity_curves(peaks_df, measure="area", azimuth_col="azimuth",
                                  hkl_col="hkl") -> dict:
    """Group labelled peaks by hkl and average their intensity per azimuth bin.

    ``measure`` is ``"area"`` (integrated, from amp/fwhm/gl) or ``"amplitude"``.
    Returns ``{hkl_label: DataFrame[azimuth, intensity, n]}`` sorted by azimuth.
    """
    out = {}
    if peaks_df is None or peaks_df.empty or hkl_col not in peaks_df.columns:
        return out
    df = peaks_df.copy()
    if measure == "area" and {"intensity", "fwhm"} <= set(df.columns):
        df["_I"] = peak_area(df["intensity"], df["fwhm"],
                             df["gl"] if "gl" in df.columns else None)
    else:
        df["_I"] = df["intensity"].astype(float)
    df = df[np.isfinite(df["_I"])]
    for hkl, sub in df.groupby(hkl_col):
        g = (sub.groupby(azimuth_col)["_I"].agg(["mean", "size"]).reset_index()
             .rename(columns={azimuth_col: "azimuth", "mean": "intensity", "size": "n"})
             .sort_values("azimuth").reset_index(drop=True))
        out[str(hkl)] = g
    return out


def build_po_model(sim_context, po_values, po_module=None):
    """Construct a ``PO.PO_Model`` from the simulation context + current PO values."""
    if po_module is None:
        import PO as po_module
    comps = [{"tau": float(po_values.get("tau", 0.0)),
              "omega": float(po_values.get("omega", 0.0)),
              "R": float(po_values.get("R", 1.0)),
              "weight": float(po_values.get("weight", 1.0))}]
    return po_module.PO_Model(
        po_model=sim_context.get("po_model") or "March-Dollase",
        components=comps,
        baseline=float(po_values.get("baseline", 0.0)),
        symmetry=sim_context["symmetry"],
        wavelength=sim_context["wavelength"],
        lattice_params=sim_context["lattice_params"],
        chi_deg=sim_context["chi"],
        POD_xtal=sim_context.get("hkl_POD") or (0, 0, 1))


def simulate_po_curves(sim_context, po_values, hkl_labels=None, n_phi=None,
                       delta=None, po_module=None) -> dict:
    """Simulated intensity vs azimuth per hkl from the PO model.

    The March-Dollase intensity is evaluated over a (phi, delta) grid and averaged over
    phi -- identical to how ``compute_strain`` forms ``Mean I @ delta`` -- then scaled by
    that hkl's structure intensity from the Simulation tab. Returns
    ``{hkl_label: (delta, intensity)}``.

    ``n_phi=None`` picks the phi sampling from R via :func:`adaptive_n_phi`, so sharply
    textured models are integrated finely enough; pass an int to force a value.
    """
    if n_phi is None:
        n_phi = adaptive_n_phi(po_values.get("R", 1.0))
    model = build_po_model(sim_context, po_values, po_module=po_module)
    if delta is None:
        delta = np.arange(-180.0, 180.0, 2.0)
    delta = np.asarray(delta, dtype=float)
    phi = np.linspace(0.0, 360.0, int(n_phi), endpoint=False)
    out = {}
    for hkl, inten in zip(sim_context["selected_hkls"], sim_context["intensities"]):
        label = hkl_label(hkl)
        if hkl_labels is not None and label not in hkl_labels:
            continue
        I_grid, _phi_grid, _delta_grid = model.intensity_for_hkl(tuple(hkl), phi, delta)
        out[label] = (delta, float(inten) * np.asarray(I_grid, dtype=float).mean(axis=0))
    return out


def _global_scale(obs_list, sim_list):
    """Closed-form least-squares scale s minimising sum((obs - s*sim)^2) over all hkls."""
    if not obs_list:
        return 1.0
    obs = np.concatenate(obs_list)
    sim = np.concatenate(sim_list)
    denom = float(np.sum(sim * sim))
    if denom <= 0 or not np.isfinite(denom):
        return 1.0
    return float(np.sum(obs * sim) / denom)


def measured_azimuth_grid(exp_curves):
    """Sorted union of the azimuths present across all hkls (the points to score at)."""
    if not exp_curves:
        return None
    return np.unique(np.concatenate([np.asarray(g["azimuth"], dtype=float)
                                     for g in exp_curves.values()]))


def evaluate_po_curves(sim_context, exp_curves, po_values, n_phi=None,
                       po_module=None, plot_delta=None) -> dict:
    """Simulate PO intensities, apply the optimal global scale, and score the match.

    Scoring uses the model evaluated EXACTLY at the measured azimuths -- interpolating a
    coarser curve onto them leaves a systematic residual that grows with texture strength
    (at R=0.2 it dominates the fit). The densely sampled curve returned for PLOTTING is a
    separate evaluation, since a smooth line needs even sampling the data does not have.

    ``n_phi=None`` selects the phi sampling adaptively from R (see :func:`adaptive_n_phi`).
    """
    n_phi = int(n_phi) if n_phi else adaptive_n_phi(po_values.get("R", 1.0))
    labels = set(exp_curves)
    # --- scoring: evaluated on the measured azimuths (queries land on grid nodes) ---
    az_eval = measured_azimuth_grid(exp_curves)
    at_data = ({} if az_eval is None else
               simulate_po_curves(sim_context, po_values, hkl_labels=labels, n_phi=n_phi,
                                  delta=az_eval, po_module=po_module))
    obs_l, sim_l = [], []
    for label, g in exp_curves.items():
        if label not in at_data:
            continue
        d, I = at_data[label]
        obs_l.append(g["intensity"].to_numpy())
        sim_l.append(interp_periodic(d, I, g["azimuth"].to_numpy()))
    scale = _global_scale(obs_l, sim_l)
    per_hkl, all_res = {}, []
    for label, g in exp_curves.items():
        if label not in at_data:
            per_hkl[label] = float("nan")
            continue
        d, I = at_data[label]
        resid = (g["intensity"].to_numpy()
                 - scale * interp_periodic(d, I, g["azimuth"].to_numpy()))
        per_hkl[label] = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
        all_res.append(resid)
    rmse = float(np.sqrt(np.mean(np.concatenate(all_res) ** 2))) if all_res else float("nan")
    # --- plotting: dense, evenly sampled curve so the model line is smooth ---
    dense = simulate_po_curves(sim_context, po_values, hkl_labels=labels, n_phi=n_phi,
                               delta=plot_delta, po_module=po_module)
    scaled = {L: (d, scale * I) for L, (d, I) in dense.items()}
    return {"sim_curves": scaled, "per_hkl": per_hkl, "rmse": rmse, "scale": scale,
            "n_phi": int(n_phi)}


def run_stage2_refinement(sim_context, exp_curves, init_values, refine_flags,
                          bounds=None, method="leastsq", max_nfev=200, n_phi=None,
                          po_module=None) -> dict:
    """Refine PO parameters (R, tau, omega, baseline) against azimuthal INTENSITIES.

    A single global scale is profiled out analytically each iteration, so the relative
    intensities between hkls constrain the fit without costing a parameter.

    ``n_phi=None`` picks the phi sampling from the STARTING R and holds it fixed: letting
    it switch tier mid-fit would step the residual and corrupt the gradients. If the
    refined R wants finer sampling, ``n_phi_suggested`` says so -- re-run to tighten.

    Adds ``scale``, ``n_phi`` and ``n_phi_suggested`` to the shared result dict.
    """
    bounds = dict(bounds or {})
    n_phi = int(n_phi) if n_phi else adaptive_n_phi(init_values.get("R", 1.0))
    labels = list(exp_curves.keys())
    az = {L: exp_curves[L]["azimuth"].to_numpy() for L in labels}
    y = {L: exp_curves[L]["intensity"].to_numpy() for L in labels}
    # Evaluate at the measured azimuths: interpolating a coarser curve onto them leaves a
    # residual floor that grows with texture strength and pulls the fit off the answer.
    az_eval = measured_azimuth_grid(exp_curves)

    def residual(values):
        sims = simulate_po_curves(sim_context, values, hkl_labels=set(labels),
                                  n_phi=n_phi, delta=az_eval, po_module=po_module)
        obs = [y[L] for L in labels if L in sims]
        mod = [interp_periodic(*sims[L], az[L]) for L in labels if L in sims]
        if not obs:
            return np.zeros(1), {"scale": 1.0}
        s = _global_scale(obs, mod)
        return np.concatenate([o - s * m for o, m in zip(obs, mod)]), {"scale": float(s)}

    return _run_refinement(
        init_values, refine_flags, residual,
        limits_fn=lambda n, v: bounds.get(n, param_bounds(n, v)),
        method=method, max_nfev=max_nfev,
        n_points=int(sum(v.size for v in y.values())),
        derived=lambda v: {"n_phi": int(n_phi),
                           "n_phi_suggested": adaptive_n_phi(v.get("R", 1.0))})


def plot_intensity_grid(exp_curves, sim_curves, ncols=4, included=None,
                        row_height=300) -> go.Figure:
    """Stage 2: intensity vs azimuth per hkl, measured against the scaled PO model."""
    return _plot_panel_grid(
        {L: (g["azimuth"].to_numpy(), g["intensity"].to_numpy(),
             *(sim_curves[L] if L in sim_curves else (None, None)))
         for L, g in exp_curves.items()},
        title="Intensity vs azimuth — data (line + points) vs PO model (dashed)",
        y_title="intensity", ncols=ncols, included=included, row_height=row_height)


# --- Stage 3: fit the background-subtracted image directly -----------------------------
#
# The image is blocked onto a regular (azimuth x 2th) grid and compared only inside a
# window around each ring; elsewhere the model is zero and the data is background, so
# including it would just dilute the residual. The model reuses the forward model of the
# validated 1D path (Generate_XRD): histogram the (phi, delta) 2th values weighted by
# intensity x PO_intensity, then convolve along 2th with the instrumental Gaussian -- so
# ring width is predicted from the strain spread plus the instrument, not fitted as a
# free shape.

STAGE3_HELP_MD = """\
### Stage 3 refinement — fit the image directly

Stages 1 and 2 fit *extracted* quantities (peak positions, then integrated intensities).
Stage 3 fits the **background-subtracted image itself**, so peak position, height and
width are all constrained at once.

**Forward model.** Identical physics to the 1D pattern simulation: for each azimuth block
the (φ, δ) orientation points are histogrammed by their 2θ, weighted by
`intensity × PO_intensity`, then convolved along 2θ with the instrumental Gaussian
(`FWHM`). The ring width is therefore *predicted* — from the strain spread across φ plus
the instrument — rather than fitted as a free shape.

**Grid.** You set the **azimuth step** and **2θ step** in degrees; both the measured image
and the model are averaged into those same boxes, so the comparison is like-for-like.
Bigger boxes average away noise but blur the peak: aim for **4–8 boxes across a ring's
FWHM**. The azimuth step is limited to **≥ 2°** because the simulation samples azimuth on
a 2° grid — finer boxes could not all be filled by the model, and the empty ones would
contribute a spurious `data − 0` residual.

**φ sampling.** The orientation average runs over φ, which in this stage does double duty:
it is the preferred-orientation integration axis *and* the source of the strain
broadening (the spread of 2θ across φ is what gives a ring its width). It is chosen
**automatically from R** — sharper texture needs more points — with a floor of 72. Cost
scales linearly with it, so it is also the main speed control. Set a non-zero value to
override. During a fit it is held fixed; if the refined R needs finer sampling you'll be
told to re-run.

**Simulation sub-sampling.** The model is *evaluated* on a finer grid than it is compared
on — `sub-samples per 2θ box` sets how much finer. The peak profile is built and convolved
at that resolution and only then averaged into the comparison boxes, so a coarse
comparison grid does not degrade the model's accuracy. Raise it if the boxes are wide
relative to the peak.

**Windows (ROI).** Only boxes within a window around each ring enter the fit. The window
is `k × (ring width)` wide, computed from the *starting* model and then **held fixed** for
the whole fit — the residual vector must keep a constant length, and a window that moved
with the parameters would change what is being compared mid-fit. The **display always
shows the full image**, with the fitted windows outlined.

**Scaling.** A single global scale is solved analytically each iteration, exactly as in
Stage 2, so relative intensities across rings and azimuths all constrain the model.

**Ring profiles.** Below the images, each ring's 2θ window is collapsed to one value per
azimuth box and plotted against azimuth, data vs model. *Integrated* (the default) sums
across the window — the ring's integrated intensity, unaffected by the peak drifting
within the window. *Peak* takes the maximum instead. The two respond differently to a
width error (too broad reads high integrated, low peak) but the same way to an intensity
error, so switching between them tells you which you are looking at.

⚠️ **Use it as a polish stage.** Seed it from Stages 1–2 and refine a few parameters at a
time. Fitted jointly, `a` trades against hydrostatic stress, and `FWHM` against intensity
and `R` — starting far away with everything free will not converge sensibly.
"""


#: Azimuth sampling of the simulation: compute_strain evaluates delta on
#: ``np.arange(-180, 180, 2)``. The comparison boxes span -180..+180 and tile it with a
#: whole number of boxes, so they stay aligned with these samples.
SIM_DELTA_STEP = 2.0
SIM_DELTA_ORIGIN = -180.0

#: Smallest usable azimuth box (degrees) -- the model cannot resolve finer than its own
#: delta sampling, so finer boxes could not all be filled.
MIN_AZ_STEP = SIM_DELTA_STEP

#: Floor on the phi sampling for Stage 3. Unlike Stage 2, phi here does double duty --
#: it is the PO integration axis AND the source of the strain broadening (the spread of
#: 2th across phi is the ring width) -- so never drop below the long-standing 72 points.
STAGE3_MIN_N_PHI = 72


@dataclass
class Stage3Grid:
    """Blocked measurement + fixed fit windows for Stage 3.

    The image is blocked ONCE over its full range; the per-ring fit windows are then just
    column spans of that same grid, so what is fitted and what is displayed are identical.

    Attributes
    ----------
    az_edges, az_centres, tth_edges, tth_centres : np.ndarray
        Block edges/centres (degrees) covering the whole image.
    data : np.ndarray
        ``(n_az, n_tth)`` block means of the measured image.
    windows : dict
        ``hkl_label -> (col_lo, col_hi)`` column span (half-open) of that ring's window.
    valid_mask : np.ndarray
        ``(n_az, n_tth)`` bool, True where the box holds usable measurement -- False over
        detector gaps, cut-off azimuth wedges and any user-excluded region. Because the
        excluded area is a per-box mask rather than an azimuth range, it is free to change
        with 2th, which is how detector shadows actually behave.
    fit_mask : np.ndarray
        ``(n_az, n_tth)`` bool, ``window_mask & valid_mask`` -- the blocks actually
        compared. Everything downstream (global scale, residual, per-hkl RMSE) keys off
        this, so masking propagates without any of them needing to know about it.
    n_sub : int
        Sub-samples per 2th block used when rendering the model before block-averaging.
    """
    az_edges: np.ndarray
    az_centres: np.ndarray
    tth_edges: np.ndarray
    tth_centres: np.ndarray
    data: np.ndarray
    windows: dict
    fit_mask: np.ndarray
    n_sub: int
    valid_mask: np.ndarray = None

    @property
    def labels(self) -> list:
        return list(self.windows.keys())

    @property
    def n_points(self) -> int:
        return int(self.fit_mask.sum())


def _wrap_into(angles, start, period=360.0):
    """Wrap angles into ``[start, start + period)`` -- azimuth is periodic."""
    return start + np.mod(np.asarray(angles, dtype=float) - start, period)


#: Columns of the Stage 3 exclusion table. Each row blanks one azimuth range over one 2th
#: range; leaving a 2th bound blank covers the whole range, so a wedge missing at all radii
#: is a single row. az_from > az_to wraps through +/-180.
EXCLUDE_REGION_COLUMNS = ["az_from", "az_to", "tth_from", "tth_to"]


def region_exclusion_mask(az_centres, tth_centres, regions) -> np.ndarray:
    """Boolean ``(n_az, n_tth)`` mask, True where a box falls in an excluded region.

    ``regions`` is an iterable of mappings with any of :data:`EXCLUDE_REGION_COLUMNS`.
    Because each row carries its own 2th range, the excluded azimuth range can differ at
    different radii -- which is what a detector shadow does. An azimuth range with
    ``az_from > az_to`` is taken to wrap through +/-180.
    """
    az = np.asarray(az_centres, dtype=float)
    tth = np.asarray(tth_centres, dtype=float)
    out = np.zeros((az.size, tth.size), dtype=bool)
    for reg in regions or ():
        get = reg.get if hasattr(reg, "get") else (lambda k, d=None: getattr(reg, k, d))
        a0, a1 = get("az_from"), get("az_to")
        t0, t1 = get("tth_from"), get("tth_to")
        if a0 is None or a1 is None or not np.isfinite(a0) or not np.isfinite(a1):
            continue
        a0, a1 = float(a0), float(a1)
        # Compare in a frame starting at a0 so a wrapping range is a simple interval.
        width = (a1 - a0) % 360.0 or 360.0
        in_az = (_wrap_into(az, a0) - a0) <= width
        t0 = -np.inf if t0 is None or not np.isfinite(t0) else float(t0)
        t1 = np.inf if t1 is None or not np.isfinite(t1) else float(t1)
        in_tth = (tth >= min(t0, t1)) & (tth <= max(t0, t1))
        out |= in_az[:, None] & in_tth[None, :]
    return out


def _ring_width_estimate(df_hkl, fwhm):
    """Expected total ring width: strain spread across phi, plus the instrument."""
    spread = 0.0
    if df_hkl is not None and len(df_hkl):
        by_delta = df_hkl.groupby("delta (degrees)")["2th"]
        rng = (by_delta.max() - by_delta.min()).to_numpy()
        if rng.size:
            spread = float(np.nanmax(rng))
    return float(np.hypot(spread, fwhm)) if spread else float(fwhm)


def build_stage3_grid(cake, grid, sim_dfs, *, az_step=5.0, tth_step=0.02, fwhm=0.1,
                      roi_k=4.0, roi_min=0.15, roi_max=2.0, n_sub=5,
                      exclude_regions=(), min_valid_frac=0.5,
                      zero_is_missing=True) -> Stage3Grid:
    """Block the whole image at the given azimuth/2th STEPS and set the fit windows.

    ``az_step`` and ``tth_step`` are spacings in degrees (not bin counts) and define the
    comparison grid for both the measurement and the model. ``sim_dfs`` (seed
    ``compute_strain`` output) only places the windows, which are then fixed.

    The azimuth boxes span the physical range -180 to +180 and are ALIGNED to the
    simulation's delta grid: a whole number of boxes tiles the period, so every box
    captures the same number of delta samples (at 2 deg exactly one each, at 10 deg five
    each) and none falls between them. ``az_step`` is clamped to :data:`MIN_AZ_STEP` --
    the model cannot resolve finer than its own delta spacing -- and then rounded to the
    nearest step that divides 360 deg evenly.

    Masking (see ``Stage3Grid.valid_mask``). Pixels that are non-finite, and by default
    exactly 0.0 (what integration software writes outside the detector -- real measured
    values are never exactly zero), count as missing; a box is dropped when fewer than
    ``min_valid_frac`` of its pixels survive. Box means are taken over the surviving
    pixels only, so a partly-covered box is not dragged toward zero. ``exclude_regions``
    additionally blanks user-specified azimuth ranges, each over its own 2th range.
    """
    twotheta = np.asarray(cake.twotheta, dtype=float)
    az = np.asarray(cake.azimuth, dtype=float)
    grid = np.asarray(grid, dtype=float)

    az_step = max(float(az_step), MIN_AZ_STEP)
    tth_step = max(float(tth_step), 1e-6)
    # Boundaries stay at +/-180; the box COUNT is what adapts to the requested step.
    n_az = max(1, int(round(360.0 / az_step)))
    az_edges = SIM_DELTA_ORIGIN + (360.0 / n_az) * np.arange(n_az + 1)
    tth_edges = np.arange(twotheta.min(), twotheta.max() + tth_step, tth_step)
    if tth_edges.size < 2:
        tth_edges = np.array([twotheta.min(), twotheta.max()])
    n_tth = tth_edges.size - 1

    row = np.clip(np.digitize(_wrap_into(az, az_edges[0]), az_edges) - 1, 0, n_az - 1)
    col = np.clip(np.digitize(twotheta, tth_edges) - 1, 0, n_tth - 1)
    good_px = np.isfinite(grid)
    if zero_is_missing:
        good_px &= grid != 0.0
    block_sum = np.zeros((n_az, n_tth))
    block_good = np.zeros((n_az, n_tth))
    block_cnt = np.zeros((n_az, n_tth))
    idx = (row[:, None], col[None, :])
    np.add.at(block_sum, idx, np.where(good_px, grid, 0.0))
    np.add.at(block_good, idx, good_px.astype(float))
    np.add.at(block_cnt, idx, np.ones_like(grid))
    # Average over the pixels that carry data, so a partly-covered box is not pulled
    # toward zero by the missing ones.
    data = np.where(block_good > 0, block_sum / np.maximum(block_good, 1.0), 0.0)

    centres_tth = 0.5 * (tth_edges[:-1] + tth_edges[1:])
    az_centres = 0.5 * (az_edges[:-1] + az_edges[1:])
    valid_mask = (block_good >= float(min_valid_frac) * np.maximum(block_cnt, 1.0))
    valid_mask &= block_good > 0
    valid_mask &= ~region_exclusion_mask(az_centres, centres_tth, exclude_regions)

    windows, fit_mask = {}, np.zeros((n_az, n_tth), dtype=bool)
    for label, df in (sim_dfs or {}).items():
        c = df.groupby("delta (degrees)")["Mean two_th @ delta"].mean().to_numpy()
        c = c[np.isfinite(c)]
        if c.size == 0:
            continue
        half = float(np.clip(roi_k * _ring_width_estimate(df, fwhm) / 2.0, roi_min, roi_max))
        lo, hi = float(c.min()) - half, float(c.max()) + half
        cols = np.where((centres_tth >= lo) & (centres_tth <= hi))[0]
        if cols.size == 0:
            continue
        windows[label] = (int(cols[0]), int(cols[-1]) + 1)
        fit_mask[:, cols[0]:cols[-1] + 1] = True
    fit_mask &= valid_mask          # masked boxes never reach the scale or the residual
    return Stage3Grid(az_edges=az_edges, az_centres=az_centres,
                      tth_edges=tth_edges, tth_centres=centres_tth, data=data,
                      windows=windows, fit_mask=fit_mask, n_sub=int(max(1, n_sub)),
                      valid_mask=valid_mask)


def render_stage3(g: Stage3Grid, sim_dfs, fwhm) -> np.ndarray:
    """Render the model onto the FULL block grid.

    Mirrors ``Generate_XRD``: each orientation point contributes
    ``intensity * PO_intensity / n_phi_at_that_delta``; the histogram is built on a
    sub-block 2th grid (``n_sub`` per block) and convolved there, then averaged down --
    convolving after blocking would under-resolve the profile. Only each ring's own 2th
    span is rendered, so cost stays independent of the full image width.
    """
    n_az, n_tth = g.data.shape
    out = np.zeros((n_az, n_tth))
    sigma = max(float(fwhm), 1e-6) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    tth_step = float(g.tth_edges[1] - g.tth_edges[0])
    step = tth_step / g.n_sub
    k_half = max(1, int(np.ceil(5.0 * sigma / step)))
    kx = np.arange(-k_half, k_half + 1) * step
    kern = np.exp(-0.5 * (kx / sigma) ** 2)
    kern /= kern.sum()

    for label, df in (sim_dfs or {}).items():
        if df is None or not len(df):
            continue
        tth = df["2th"].to_numpy(dtype=float)
        delta = df["delta (degrees)"].to_numpy(dtype=float)
        w = (df["intensity"].to_numpy(dtype=float)
             * df["PO_intensity"].to_numpy(dtype=float))
        _, inv, cnt = np.unique(delta, return_inverse=True, return_counts=True)
        w = w / cnt[inv]
        # Render over this ring's own span, padded so the convolution tails are included.
        pad = k_half * step + tth_step
        lo = max(float(np.nanmin(tth)) - pad, float(g.tth_edges[0]))
        hi = min(float(np.nanmax(tth)) + pad, float(g.tth_edges[-1]))
        c0 = int(np.clip(np.searchsorted(g.tth_edges, lo) - 1, 0, n_tth - 1))
        c1 = int(np.clip(np.searchsorted(g.tth_edges, hi), c0 + 1, n_tth))
        n_cols = c1 - c0
        fine0 = float(g.tth_edges[c0])
        n_fine = n_cols * g.n_sub
        # Split each delta between the two azimuth boxes whose CENTRES bracket it, weighted
        # by distance (the bilinear convention cake_dict_to_2Dcake uses). Dropping a delta
        # into whichever box merely contains it would place the model at a box edge, a
        # half-box phase error against the data -- which is a box mean. Wrapping the index
        # keeps the +/-180 seam continuous.
        az_box = 360.0 / n_az
        centre0 = float(g.az_edges[0]) + 0.5 * az_box
        pos = (_wrap_into(delta, centre0) - centre0) / az_box
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        i0m, i1m = i0 % n_az, (i0 + 1) % n_az

        f_idx = np.floor((tth - fine0) / step).astype(int)
        ok = (f_idx >= 0) & (f_idx < n_fine)
        if not ok.any():
            continue
        hist = np.zeros((n_az, n_fine))
        np.add.at(hist, (i0m[ok], f_idx[ok]), w[ok] * (1.0 - frac[ok]))
        np.add.at(hist, (i1m[ok], f_idx[ok]), w[ok] * frac[ok])
        # Measured blocks are MEANS over their pixels, so divide by the effective number of
        # delta samples reaching each box, accumulated with the same weights.
        _, first = np.unique(delta, return_index=True)
        wsum = np.zeros(n_az)
        np.add.at(wsum, i0m[first], 1.0 - frac[first])
        np.add.at(wsum, i1m[first], frac[first])
        hist /= np.maximum(wsum, 1e-9)[:, None]
        conv = fftconvolve(hist, kern[None, :], mode="same", axes=1)
        # Sum the fine samples in each box and divide by the box width, giving an
        # intensity DENSITY. Taking the mean instead would make the model amplitude scale
        # with n_sub -- absorbed by the global scale, but it would leave the model values
        # dependent on an internal sampling choice.
        out[:, c0:c1] += conv.reshape(n_az, n_cols, g.n_sub).sum(axis=2) / tth_step
    return out


def stage3_n_phi(values) -> int:
    """Phi sampling for Stage 3: adaptive in R, floored at :data:`STAGE3_MIN_N_PHI`."""
    return max(STAGE3_MIN_N_PHI, adaptive_n_phi((values or {}).get("R", 1.0)))


def _stage3_sim_dfs(strain_fn, sim_context, values, coarse=False, n_phi=None,
                    po_apply=None):
    """compute_strain DataFrames for every hkl at the given parameter values.

    ``n_phi=None`` picks the phi sampling from R via :func:`stage3_n_phi`.

    ``po_apply`` is a callback invoked with ``values`` before the forward model runs.
    It is REQUIRED to refine PO parameters: compute_strain reads R/tau/omega/weight/
    baseline from Streamlit session state rather than from its arguments, so without this
    hook those parameters never reach the simulation and the optimiser reports that they
    "did not affect the fit". The callback is injected so this module stays UI-free.
    """
    if po_apply is not None:
        po_apply(values)
    lattice_params, sigma_params, chi = _assemble_params(sim_context, values)
    n_phi = int(n_phi) if n_phi else stage3_n_phi(values)
    phi_values = np.radians(np.linspace(0.0, 360.0, n_phi, endpoint=False))
    psi_values = 1 if coarse else 0
    out = {}
    for hkl, inten in zip(sim_context["selected_hkls"], sim_context["intensities"]):
        label, df, _psi, _s = strain_fn(
            hkl, inten, sim_context["symmetry"], lattice_params, sim_context["wavelength"],
            sim_context["cijs"], sigma_params, chi, phi_values, psi_values,
            po_model=sim_context.get("po_model"))
        out[label] = df
    return out


def evaluate_stage3(strain_fn, sim_context, g: Stage3Grid, values, fwhm,
                    coarse=False, n_phi=None, po_apply=None) -> dict:
    """Render the model on the full block grid, scale it, and score it inside the windows.

    ``sim`` is returned over the WHOLE grid (for display); the scale and residuals use only
    the blocks in ``g.fit_mask``. ``n_phi=None`` selects the phi sampling from R.
    """
    n_phi = int(n_phi) if n_phi else stage3_n_phi(values)
    sim_dfs = _stage3_sim_dfs(strain_fn, sim_context, values, coarse=coarse, n_phi=n_phi,
                              po_apply=po_apply)
    sim = render_stage3(g, sim_dfs, fwhm)
    m = g.fit_mask
    scale = _global_scale([g.data[m]], [sim[m]]) if m.any() else 1.0
    sim = scale * sim
    per_hkl = {}
    for label, (c0, c1) in g.windows.items():
        win = g.fit_mask[:, c0:c1]
        resid = (g.data[:, c0:c1] - sim[:, c0:c1])[win]
        per_hkl[label] = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
    rmse = (float(np.sqrt(np.mean((g.data[m] - sim[m]) ** 2))) if m.any() else float("nan"))
    return {"sim": sim, "per_hkl": per_hkl, "rmse": rmse, "scale": scale,
            "n_points": g.n_points, "n_phi": int(n_phi)}


def run_stage3_refinement(strain_fn, sim_context, g: Stage3Grid, init_values,
                          refine_flags, bounds=None, method="leastsq", max_nfev=200,
                          coarse=False, lattice_frac=LATTICE_BOUND_FRAC,
                          n_phi=None, po_apply=None) -> dict:
    """Refine against the blocked IMAGE; ``fwhm`` participates like any other parameter.

    The grid and its ``fit_mask`` are fixed, so the residual keeps a constant length --
    required by the least-squares minimisers. ``n_phi`` is likewise fixed from the
    starting R (see :func:`run_stage2_refinement`), and ``po_apply`` is required for PO
    parameters to reach the model at all (see :func:`_stage3_sim_dfs`).

    Adds ``scale``, ``n_phi`` and ``n_phi_suggested`` to the shared result dict.
    """
    bounds = dict(bounds or {})
    n_phi = int(n_phi) if n_phi else stage3_n_phi(init_values)

    def residual(values):
        ev = evaluate_stage3(strain_fn, sim_context, g, values,
                             values.get("fwhm", 0.1), coarse=coarse, n_phi=n_phi,
                             po_apply=po_apply)
        return (g.data[g.fit_mask] - ev["sim"][g.fit_mask]), {"scale": ev["scale"]}

    return _run_refinement(
        init_values, refine_flags, residual,
        limits_fn=lambda n, v: bounds.get(n, param_bounds(n, v, lattice_frac)),
        method=method, max_nfev=max_nfev, n_points=g.n_points,
        derived=lambda v: {"n_phi": int(n_phi), "n_phi_suggested": stage3_n_phi(v)})


def _nudge_colour(colour):
    """Return a visually identical colour that is not byte-identical to ``colour``."""
    try:
        if isinstance(colour, str) and colour.startswith("#") and len(colour) == 7:
            rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
        elif isinstance(colour, str) and colour.strip().lower().startswith("rgb"):
            rgb = [int(float(v)) for v in colour[colour.index("(") + 1:
                                                 colour.index(")")].split(",")[:3]]
        else:
            return colour
    except Exception:
        return colour
    rgb[0] = rgb[0] + 1 if rgb[0] < 255 else rgb[0] - 1
    return "#%02x%02x%02x" % tuple(max(0, min(255, v)) for v in rgb)


def stage3_azimuthal_profiles(g: Stage3Grid, sim, measure="integrated") -> dict:
    """Collapse each ring's 2th window to one value per azimuth box.

    ``measure``:

    - ``"integrated"`` (default) -- sum across the window times the box width, i.e. the
      ring's integrated intensity at that azimuth. This is the physically meaningful
      quantity, it is what Stage 2 fits, and it is unaffected by the peak drifting within
      the window as strain shifts it.
    - ``"peak"`` -- the maximum across the window (ring height). The two respond
      differently to a width error: too broad a model sits high on the integrated curve
      and low on the peak curve (measured 1.19x vs 0.78x for a 60% FWHM error), whereas a
      plain intensity error moves both the same way. Comparing them therefore tells the
      two apart. They are not independent of the global scale, which refits either way.

    Returns ``{hkl_label: DataFrame[azimuth, data, model]}``. ``sim`` is expected already
    scaled (as :func:`evaluate_stage3` returns it), so the two are directly comparable.
    """
    sim = np.asarray(sim, dtype=float)
    tth_step = float(g.tth_edges[1] - g.tth_edges[0]) if g.tth_edges.size > 1 else 1.0
    out = {}
    for label, (c0, c1) in g.windows.items():
        # Masked boxes are dropped, not counted as zero, so a partly-blanked azimuth is
        # not shown as a dip. An azimuth with nothing left becomes NaN and the line breaks.
        keep = g.fit_mask[:, c0:c1]
        d = np.where(keep, g.data[:, c0:c1], np.nan)
        m = np.where(keep, sim[:, c0:c1], np.nan)
        any_left = keep.any(axis=1)
        with np.errstate(invalid="ignore"):
            if measure == "peak":
                dv, sv = np.nanmax(d, axis=1), np.nanmax(m, axis=1)
            else:
                # Scale by the full window width so partly-masked azimuths stay comparable
                # with fully-covered ones rather than reading low.
                n_win = max(c1 - c0, 1)
                dv = np.nanmean(d, axis=1) * n_win * tth_step
                sv = np.nanmean(m, axis=1) * n_win * tth_step
        dv = np.where(any_left, dv, np.nan)
        sv = np.where(any_left, sv, np.nan)
        out[label] = pd.DataFrame({"azimuth": g.az_centres, "data": dv, "model": sv})
    return out


def plot_stage3_azimuthal(profiles, ncols=4, row_height=280,
                          measure="integrated") -> go.Figure:
    """Stage 3: each ring's collapsed 2th window vs azimuth, data against model."""
    ylab = "peak intensity" if measure == "peak" else "integrated intensity"
    return _plot_panel_grid(
        {L: (p["azimuth"].to_numpy(), p["data"].to_numpy(),
             p["azimuth"].to_numpy(), p["model"].to_numpy())
         for L, p in profiles.items()},
        title=f"Ring {ylab} vs azimuth — data (line + points) vs model (dashed)",
        y_title=ylab, ncols=ncols, row_height=row_height)


def _explicit_colorscale(name):
    """Resolve a named colorscale to an explicit ``[[pos, colour], ...]`` list.

    Streamlit rewrites the darkest stop of a heatmap colorscale on its way to the browser
    (Inferno's ``#000004`` arrives as the theme accent, or as a Plotly default under
    ``theme=None``), turning the dark background these images rely on into a block of
    colour. The substitution matches on the COLOUR VALUE, so the first stop is nudged by a
    single 8-bit step -- indistinguishable to the eye, but no longer a match.
    """
    try:
        import plotly.colors as _pc
        cs = [[float(p), c] for p, c in _pc.get_colorscale(name)]
    except Exception:
        return name
    if cs:
        cs[0] = [cs[0][0], _nudge_colour(cs[0][1])]
    return cs


def plot_stage3_comparison(g: Stage3Grid, sim, percentile=99.5, row_height=300,
                           colorscale="Inferno", diff_colorscale="RdBu",
                           show_windows=True, tth_range=None) -> go.Figure:
    """Full-image data / model / residual, stacked and zoom-linked.

    Three rows over the WHOLE 2th range (not just the fitted windows) so the model can be
    judged in context. The x axes are shared, so a click-drag zoom on any panel zooms all
    three together. Data and model share one intensity scale (``colorscale``, clipped at
    ``percentile`` for contrast); the residual uses a symmetric diverging scale centred on
    zero. Fitted windows are outlined when ``show_windows``.
    """
    data = np.asarray(g.data, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if g.valid_mask is not None:
        # NaN renders transparent, so masked regions read as holes rather than as zeros.
        blank = ~np.asarray(g.valid_mask, dtype=bool)
        data = np.where(blank, np.nan, data)
        sim = np.where(blank, np.nan, sim)
    finite = data[np.isfinite(data)]
    zmax = float(np.nanpercentile(finite, percentile)) if finite.size else 1.0
    if not np.isfinite(zmax) or zmax <= 0:
        zmax = float(np.nanmax(finite)) if finite.size else 1.0
    diff = data - sim
    dfin = diff[np.isfinite(diff)]
    dmax = float(np.nanpercentile(np.abs(dfin), percentile)) if dfin.size else 1.0
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = zmax

    seq = _explicit_colorscale(colorscale)
    div = _explicit_colorscale(diff_colorscale)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Measured (block-averaged)", "Simulated", "Difference (data − model)"),
        vertical_spacing=0.06)
    x, y = g.tth_centres, g.az_centres
    for r, (z, cs, zl, zh, cbar_y) in enumerate([
            (data, seq, 0.0, zmax, 0.86),
            (sim, seq, 0.0, zmax, 0.5),
            (diff, div, -dmax, dmax, 0.14)]):
        fig.add_trace(go.Heatmap(
            z=z, x=x, y=y, colorscale=cs, zmin=zl, zmax=zh, zsmooth=False,
            colorbar=dict(len=0.28, y=cbar_y, thickness=12),
            hovertemplate="2θ %{x:.3f}°<br>azimuth %{y:.1f}°<br>%{z:.4g}<extra></extra>"),
            row=r + 1, col=1)
    if show_windows:
        for (c0, c1) in g.windows.values():
            lo = float(g.tth_edges[c0])
            hi = float(g.tth_edges[min(c1, g.tth_edges.size - 1)])
            for r in (1, 2, 3):
                fig.add_vrect(x0=lo, x1=hi, line_width=1, line_color="#00e5ff",
                              fillcolor="rgba(0,0,0,0)", row=r, col=1)
    fig.update_xaxes(title_text="2θ (degrees)", row=3, col=1)
    for r in (1, 2, 3):
        fig.update_yaxes(title_text="azimuth (°)", row=r, col=1)
    if tth_range is not None:
        fig.update_xaxes(range=list(tth_range))
    fig.update_layout(height=row_height * 3, margin=dict(l=70, r=20, t=60, b=50),
                      title="Stage 3 — measured vs simulated vs difference",
                      dragmode="zoom")
    return fig
