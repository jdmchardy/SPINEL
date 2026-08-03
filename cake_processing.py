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
from scipy.signal import find_peaks


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
# Stage 1 refinement: match simulated Mean two_th @ delta to experimental
# peak positions (2th vs azimuth), per hkl reflection.
#
# The forward model is spinel_core.compute_strain, INJECTED as ``strain_fn`` so
# this module never imports spinel_core (which pulls in pyFAI/lmfit and imports
# cake_processing itself -> would be circular). The optimiser is
# scipy.optimize.least_squares, so refinement runs anywhere scipy is available.
# =========================================================================

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


#: Default refinement limits. Lattice lengths are bounded RELATIVE to the starting value
#: (a window of +/- LATTICE_BOUND_FRAC around it), so a poor initial guess restricts how
#: far the refinement can travel -- widen the fraction, or the explicit min/max in the UI,
#: when starting far from the answer. Stress/chi use absolute physical ranges.
LATTICE_BOUND_FRAC = 0.25
SIGMA_BOUNDS = (-25.0, 25.0)
CHI_BOUNDS = (-90.0, 90.0)


def default_param_bounds(name, value, lattice_frac=LATTICE_BOUND_FRAC):
    """Default (min, max) refinement limits for a parameter.

    Lattice lengths get ``value * (1 -/+ lattice_frac)`` — a window centred on the START
    value, which is why a far-off initial guess can clamp the refinement. Stress components
    use ``SIGMA_BOUNDS`` (GPa) and chi uses ``CHI_BOUNDS`` (degrees).
    """
    if name in ("a_val", "b_val", "c_val"):
        if value:
            return ((1.0 - lattice_frac) * value, (1.0 + lattice_frac) * value)
        return (0.0, np.inf)
    if name.startswith("sigma_"):
        return SIGMA_BOUNDS
    if name == "chi":
        return CHI_BOUNDS
    return (-np.inf, np.inf)


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
    """Refine peak positions across the included hkls with lmfit.

    ``init_values`` holds a starting value for every refineable parameter (lattice
    a/b/c, the six sigma components, chi); ``refine_flags`` selects which vary.
    ``bounds`` optionally overrides the per-parameter ``(min, max)`` limits (see
    :func:`default_param_bounds` — note the lattice window is relative to the START
    value, so a far-off initial guess restricts the reachable range).

    Falls back to ``scipy.optimize.least_squares`` if lmfit is unavailable.

    Returns a dict with ``values`` (refined, full set), ``errors``, ``success``,
    ``message``, ``rmse``, ``n_points``, ``n_free``, ``t``, ``report`` (lmfit fit
    report text), ``at_limit`` (params sitting on a bound) and fit statistics.
    """
    free = [n for n, on in refine_flags.items() if on and n in init_values]
    labels = list(exp_curves.keys())
    az = {L: exp_curves[L]["azimuth"].to_numpy() for L in labels}
    y = {L: exp_curves[L]["mean_2th"].to_numpy() for L in labels}
    n_points = int(sum(v.size for v in y.values()))
    bounds = dict(bounds or {})

    def residuals_from_values(values):
        lattice_params, sigma_params, chi = _assemble_params(sim_context, values)
        sims = simulate_all_curves(strain_fn, sim_context, lattice_params, sigma_params,
                                   chi, hkl_labels=set(labels), coarse=coarse)
        res = []
        for L in labels:
            if L not in sims:
                continue
            delta, sim2th = sims[L]
            res.append(y[L] - interp_periodic(delta, sim2th, az[L]))
        return np.concatenate(res) if res else np.zeros(1)

    def _package(refined, errors, success, message, resid, report="", at_limit=None,
                 stats=None):
        out = {"values": refined, "errors": errors, "success": bool(success),
               "message": str(message),
               "rmse": float(np.sqrt(np.mean(np.asarray(resid) ** 2))),
               "n_points": n_points, "n_free": len(free),
               "t": float(refined.get("sigma_33", 0.0) - refined.get("sigma_11", 0.0)),
               "report": report, "at_limit": at_limit or [], "method": method}
        out.update(stats or {})
        return out

    if not free:
        return _package(dict(init_values), {}, True, "No parameters selected to refine.",
                        residuals_from_values(dict(init_values)))

    limits = {n: bounds.get(n, default_param_bounds(n, init_values[n], lattice_frac))
              for n in free}

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
            return residuals_from_values(values)

        result = lm_minimize(lm_residual, lm_params, method=method, max_nfev=max_nfev)
        refined = dict(init_values)
        errors, at_limit = {}, []
        for n in free:
            par = result.params[n]
            refined[n] = float(par.value)
            errors[n] = float(par.stderr) if par.stderr is not None else float("nan")
            lo, hi = limits[n]
            if np.isfinite(lo) and np.isclose(par.value, lo, rtol=1e-6, atol=1e-9):
                at_limit.append(f"{n} (at min {lo:g})")
            elif np.isfinite(hi) and np.isclose(par.value, hi, rtol=1e-6, atol=1e-9):
                at_limit.append(f"{n} (at max {hi:g})")
        stats = {"chisqr": float(getattr(result, "chisqr", np.nan)),
                 "redchi": float(getattr(result, "redchi", np.nan)),
                 "aic": float(getattr(result, "aic", np.nan)),
                 "bic": float(getattr(result, "bic", np.nan)),
                 "nfev": int(getattr(result, "nfev", 0))}
        return _package(refined, errors, getattr(result, "success", True),
                        getattr(result, "message", ""), result.residual,
                        report=lm_fit_report(result), at_limit=at_limit, stats=stats)

    # --- Fallback: scipy least_squares (lmfit not installed) ---
    def residuals_vec(vec):
        values = dict(init_values)
        values.update({n: v for n, v in zip(free, vec)})
        return residuals_from_values(values)

    lo = np.array([limits[n][0] for n in free], dtype=float)
    hi = np.array([limits[n][1] for n in free], dtype=float)
    x0 = np.clip(np.array([init_values[n] for n in free], dtype=float), lo, hi)
    result = least_squares(residuals_vec, x0, bounds=(lo, hi), max_nfev=max_nfev)
    refined = dict(init_values)
    refined.update({n: float(v) for n, v in zip(free, result.x)})
    return _package(refined, _stderr_from_result(result, free), result.success,
                    result.message, result.fun,
                    report="lmfit not installed — used scipy.optimize.least_squares.")


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


def plot_refinement_grid(exp_curves, sim_curves, ncols=4, included=None,
                         row_height=300) -> go.Figure:
    """Grid (``ncols`` wide) of 2θ-vs-azimuth panels: data points + simulated line.

    Excluded hkls (not in ``included``) are greyed. One panel per hkl in ``exp_curves``.
    """
    labels = list(exp_curves.keys())
    n = len(labels)
    if n == 0:
        return go.Figure()
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(n / ncols))
    included = set(labels) if included is None else set(included)
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"hkl {L}" for L in labels],
                        horizontal_spacing=0.06,
                        vertical_spacing=_subplot_vspace(nrows, row_height))
    for i, L in enumerate(labels):
        r, c = i // ncols + 1, i % ncols + 1
        on = L in included
        g = exp_curves[L]
        fig.add_trace(go.Scatter(
            x=g["azimuth"], y=g["mean_2th"], mode="markers", name=f"{L} data",
            marker=dict(color="#e6194B" if on else "#c9c9c9", size=5),
            showlegend=False), row=r, col=c)
        if L in sim_curves:
            delta, sim2th = sim_curves[L]
            o = np.argsort(np.asarray(delta, dtype=float))
            fig.add_trace(go.Scatter(
                x=np.asarray(delta)[o], y=np.asarray(sim2th)[o], mode="lines",
                name=f"{L} sim",
                line=dict(color="#4363d8" if on else "#dddddd", width=1.6),
                showlegend=False), row=r, col=c)
    fig.update_xaxes(title_text="azimuth (°)", title_standoff=6)
    fig.update_yaxes(title_text="2θ (°)", title_standoff=6)
    fig.update_layout(height=row_height * nrows, margin=dict(l=60, r=20, t=70, b=50),
                      title="2θ vs azimuth — data (points) vs simulation (line)")
    return fig


# =========================================================================
# Stage 2 refinement: preferred-orientation (PO) parameters from the
# INTENSITY variation with azimuth, per hkl.
#
# Experimental intensity = integrated peak AREA (from the stored amp/fwhm/gl, so it is
# unbiased when peak width varies with azimuth). Simulated intensity comes straight from
# PO.PO_Model.intensity_for_hkl averaged over phi -- the same quantity compute_strain
# builds into "Mean I @ delta" -- scaled by each hkl's structure intensity, then matched
# to the data through a SINGLE GLOBAL scale solved in closed form each iteration (so the
# relative intensities between hkls still constrain the model, at no parameter cost).
# =========================================================================

#: PO parameters Stage 2 can refine, with (min, max) limits.
PO_PARAM_BOUNDS = {"R": (0.05, 5.0), "tau": (-180.0, 180.0),
                   "omega": (-180.0, 180.0), "baseline": (0.0, 1.0)}

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


def simulate_po_curves(sim_context, po_values, hkl_labels=None, n_phi=36,
                       delta=None, po_module=None) -> dict:
    """Simulated intensity vs azimuth per hkl from the PO model.

    The March-Dollase intensity is evaluated over a (phi, delta) grid and averaged over
    phi -- identical to how ``compute_strain`` forms ``Mean I @ delta`` -- then scaled by
    that hkl's structure intensity from the Simulation tab. Returns
    ``{hkl_label: (delta, intensity)}``.
    """
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


def evaluate_po_curves(sim_context, exp_curves, po_values, n_phi=36,
                       po_module=None) -> dict:
    """Simulate PO intensities, apply the optimal global scale, and score the match."""
    sims = simulate_po_curves(sim_context, po_values, hkl_labels=set(exp_curves),
                              n_phi=n_phi, po_module=po_module)
    obs_l, sim_l = [], []
    for label, g in exp_curves.items():
        if label not in sims:
            continue
        d, I = sims[label]
        obs_l.append(g["intensity"].to_numpy())
        sim_l.append(interp_periodic(d, I, g["azimuth"].to_numpy()))
    scale = _global_scale(obs_l, sim_l)
    scaled = {L: (d, scale * I) for L, (d, I) in sims.items()}
    per_hkl, all_res = {}, []
    for label, g in exp_curves.items():
        if label not in scaled:
            per_hkl[label] = float("nan")
            continue
        d, I = scaled[label]
        resid = g["intensity"].to_numpy() - interp_periodic(d, I, g["azimuth"].to_numpy())
        per_hkl[label] = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
        all_res.append(resid)
    rmse = float(np.sqrt(np.mean(np.concatenate(all_res) ** 2))) if all_res else float("nan")
    return {"sim_curves": scaled, "per_hkl": per_hkl, "rmse": rmse, "scale": scale}


def run_stage2_refinement(sim_context, exp_curves, init_values, refine_flags,
                          bounds=None, method="leastsq", max_nfev=200, n_phi=36,
                          po_module=None) -> dict:
    """Refine PO parameters (R, tau, omega, baseline) against azimuthal intensities.

    A single global scale factor is profiled out analytically at every iteration, so the
    relative intensities between hkls constrain the fit without costing a parameter.
    Returns the same shape of result dict as :func:`run_stage1_refinement`, plus
    ``scale``.
    """
    free = [n for n, on in refine_flags.items() if on and n in init_values]
    labels = list(exp_curves.keys())
    az = {L: exp_curves[L]["azimuth"].to_numpy() for L in labels}
    y = {L: exp_curves[L]["intensity"].to_numpy() for L in labels}
    n_points = int(sum(v.size for v in y.values()))
    bounds = dict(bounds or {})
    limits = {n: bounds.get(n, PO_PARAM_BOUNDS.get(n, (-np.inf, np.inf))) for n in free}

    def residuals_from_values(values):
        sims = simulate_po_curves(sim_context, values, hkl_labels=set(labels),
                                  n_phi=n_phi, po_module=po_module)
        obs_l, sim_l, order = [], [], []
        for L in labels:
            if L not in sims:
                continue
            d, I = sims[L]
            obs_l.append(y[L])
            sim_l.append(interp_periodic(d, I, az[L]))
            order.append(L)
        if not obs_l:
            return np.zeros(1), 1.0
        s = _global_scale(obs_l, sim_l)
        return np.concatenate([o - s * m for o, m in zip(obs_l, sim_l)]), s

    def _package(values, errors, success, message, resid, scale, report="",
                 at_limit=None, stats=None):
        out = {"values": values, "errors": errors, "success": bool(success),
               "message": str(message),
               "rmse": float(np.sqrt(np.mean(np.asarray(resid) ** 2))),
               "n_points": n_points, "n_free": len(free), "scale": float(scale),
               "report": report, "at_limit": at_limit or [], "method": method}
        out.update(stats or {})
        return out

    if not free:
        r, s = residuals_from_values(dict(init_values))
        return _package(dict(init_values), {}, True,
                        "No parameters selected to refine.", r, s)

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
            return residuals_from_values(values)[0]

        result = lm_minimize(lm_residual, lm_params, method=method, max_nfev=max_nfev)
        refined = dict(init_values)
        errors, at_limit = {}, []
        for n in free:
            par = result.params[n]
            refined[n] = float(par.value)
            errors[n] = float(par.stderr) if par.stderr is not None else float("nan")
            lo, hi = limits[n]
            if np.isfinite(lo) and np.isclose(par.value, lo, rtol=1e-6, atol=1e-9):
                at_limit.append(f"{n} (at min {lo:g})")
            elif np.isfinite(hi) and np.isclose(par.value, hi, rtol=1e-6, atol=1e-9):
                at_limit.append(f"{n} (at max {hi:g})")
        _r, _s = residuals_from_values(refined)
        stats = {"chisqr": float(getattr(result, "chisqr", np.nan)),
                 "redchi": float(getattr(result, "redchi", np.nan)),
                 "aic": float(getattr(result, "aic", np.nan)),
                 "bic": float(getattr(result, "bic", np.nan)),
                 "nfev": int(getattr(result, "nfev", 0))}
        return _package(refined, errors, getattr(result, "success", True),
                        getattr(result, "message", ""), _r, _s,
                        report=lm_fit_report(result), at_limit=at_limit, stats=stats)

    # --- Fallback: scipy least_squares ---
    lo = np.array([limits[n][0] for n in free], dtype=float)
    hi = np.array([limits[n][1] for n in free], dtype=float)
    x0 = np.clip(np.array([init_values[n] for n in free], dtype=float), lo, hi)

    def residuals_vec(vec):
        values = dict(init_values)
        values.update({n: v for n, v in zip(free, vec)})
        return residuals_from_values(values)[0]

    result = least_squares(residuals_vec, x0, bounds=(lo, hi), max_nfev=max_nfev)
    refined = dict(init_values)
    refined.update({n: float(v) for n, v in zip(free, result.x)})
    _r, _s = residuals_from_values(refined)
    return _package(refined, _stderr_from_result(result, free), result.success,
                    result.message, _r, _s,
                    report="lmfit not installed — used scipy.optimize.least_squares.")


def plot_intensity_grid(exp_curves, sim_curves, ncols=4, included=None,
                        row_height=300) -> go.Figure:
    """Grid of intensity-vs-azimuth panels: data points + scaled simulated PO curve."""
    labels = list(exp_curves.keys())
    n = len(labels)
    if n == 0:
        return go.Figure()
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(n / ncols))
    included = set(labels) if included is None else set(included)
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"hkl {L}" for L in labels],
                        horizontal_spacing=0.06,
                        vertical_spacing=_subplot_vspace(nrows, row_height))
    for i, L in enumerate(labels):
        r, c = i // ncols + 1, i % ncols + 1
        on = L in included
        g = exp_curves[L]
        fig.add_trace(go.Scatter(
            x=g["azimuth"], y=g["intensity"], mode="markers", name=f"{L} data",
            marker=dict(color="#e6194B" if on else "#c9c9c9", size=5),
            showlegend=False), row=r, col=c)
        if L in sim_curves:
            d, I = sim_curves[L]
            o = np.argsort(np.asarray(d, dtype=float))
            fig.add_trace(go.Scatter(
                x=np.asarray(d)[o], y=np.asarray(I)[o], mode="lines", name=f"{L} sim",
                line=dict(color="#4363d8" if on else "#dddddd", width=1.6),
                showlegend=False), row=r, col=c)
    fig.update_xaxes(title_text="azimuth (°)", title_standoff=6)
    fig.update_yaxes(title_text="intensity", title_standoff=6)
    fig.update_layout(height=row_height * nrows, margin=dict(l=60, r=20, t=70, b=50),
                      title="Intensity vs azimuth — data (points) vs PO model (line)")
    return fig
