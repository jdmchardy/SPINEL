import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import io
import pyFAI
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from lmfit import Parameters, minimize, fit_report
from pyFAI import AzimuthalIntegrator
import tempfile
import zipfile

#For interactive plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
from pathlib import Path

#Import personal modules
import PO #Preferred Orientation Model
import cake_processing as cp #Dioptas cake (.txt) import/processing

#3d plotting
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import RegularGridInterpolator
from spinel_core import (Gaussian, stress_tensor_to_voigt, voigt_to_strain_tensor,
                         get_d0, get_elastic, cake_dict_to_2Dcake, compute_bin_indices,
                         generate_1D_XRD_plot, generate_1D_XRD_overlay, store_download,
                         compute_strain, Generate_XRD, batch_XRD, cake_data,
                         run_refinement, cost_function, generate_epsilon_psi_curves,
                         setup_refinement_toggles, generate_cake_figures)

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 12px !important;   /* Adjust this value to your desired size */
}

/* Smaller widget labels */
label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    font-size: 12px !important;
}

/* Smaller number + text input text */
input, textarea, select {
    font-size: 12px !important;
}

/* Smaller checkbox labels */
.stCheckbox label {
    font-size: 12px !important;
}

/* Smaller markdown text */
p, span, div {
    font-size: 12px !important;
}

/* Make headers smaller too */
h1, h2, h3, h4, h5 {
    font-size: 18px !important;
}

/* Reduce vertical gaps between all widgets */
.stNumberInput, .stTextInput, .stSelectbox, .stSlider, .stCheckbox {
    margin-top: 0.1rem !important;
    margin-bottom: 0.1rem !important;
}

/* Reduce extra padding around Streamlit containers */
div[data-testid="stVerticalBlock"] {
    gap: 0.1rem !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
#### Main App -----------------------------------------------------
# -----------------------------------------------------------------------
    
st.set_page_config(layout="wide")

BASE_DIR = Path(__file__).parent
logo_path = BASE_DIR / "spinel_logo.png"

img = Image.open(logo_path)

col_img, col_title = st.columns([1, 3])

with col_img:
    st.image(img, width='stretch')


tab_sim, tab_cake, tab_peaks, tab_refine = st.tabs(
    ["Simulation", "Cake Import & Background", "Peak Extraction", "Refinement"])

with tab_cake:
    # --- Cake Import (independent of the elastic/hkl CSV workflow) ---
    # Loads a Dioptas 2D-integration .txt "cake" export and plots it as an
    # interactive heatmap. Available without uploading any other files; later
    # processing phases reuse st.session_state.imported_cake.
    st.subheader("Cake Import (Dioptas .txt)")
    cake_col1, cake_col2 = st.columns([1, 3])
    with cake_col1:
        cake_file = st.file_uploader("Dioptas cake .txt", type=["txt"], key="cake_txt")
        cake_percentile = st.slider(
            "Display contrast (percentile)",
            min_value=90.0, max_value=100.0, value=99.5, step=0.1, format="%.1f",
            help="Upper display clip set to this percentile of pixel intensities "
                 "(auto-adapts per file). Higher = darker (clips fewer bright "
                 "pixels); lower = brighter faint rings.",
        )

    if cake_file is not None:
        try:
            cake = cp.load_cake_data(cake_file, filename=cake_file.name)
        except ValueError as e:
            st.error(f"Failed to load cake file: {e}")
        else:
            st.session_state.imported_cake = cake
            with cake_col2:
                st.plotly_chart(
                    cp.plot_cake_heatmap(cake, percentile=cake_percentile),
                    width='stretch',
                )
                st.caption(
                    "Grid {}×{} (azimuth×2θ) · 2θ {:.2f}–{:.2f}° · azimuth {:.1f}–{:.1f}°".format(
                        cake.intensity.shape[0], cake.intensity.shape[1],
                        float(cake.twotheta.min()), float(cake.twotheta.max()),
                        float(cake.azimuth.min()), float(cake.azimuth.max()),
                    )
                )

    # --- Background Subtraction (per-azimuth Chebyshev polynomial) ---
    # Builds a 2D background image by fitting a polynomial to the peak-masked, gap-
    # filled background points of each azimuth row, then subtracts it. Result is
    # stored in st.session_state.cake_background for later phases.
    if st.session_state.get("imported_cake") is not None:
        st.subheader("Background Subtraction")
        _cake = st.session_state.imported_cake

        with st.expander("ℹ️ How the background subtraction works"):
            st.markdown(cp.BACKGROUND_HELP_MD)

        # Azimuth binning drives the background: rows are averaged per bin, the fit is
        # done on the binned profile, then applied to all rows in the bin. Kept outside
        # the form so the effective bin width updates live.
        _n_az = int(_cake.azimuth.size)
        n_az_bins = st.number_input(
            "Number of azimuth bins", min_value=1, max_value=_n_az,
            value=max(1, _n_az // 2), step=1,
            help="Azimuth rows are grouped into this many equal bins. Within a bin the "
                 "rows are averaged, ONE background is fitted to that averaged profile, "
                 "and it is subtracted from every (finer-resolution) row in the bin. "
                 "Cannot exceed the number of azimuth rows in the data ({}); default is "
                 "half of that. Fewer bins → cleaner profile to fit but coarser azimuthal "
                 "variation; more (finer) bins usually track azimuthal variation "
                 "better.".format(_n_az))
        _bin_width = cp.assign_azimuth_bins(_cake.azimuth, int(n_az_bins))[2]
        st.caption(
            f"Effective azimuth binning: {_bin_width:.2f}° per bin "
            f"({int(n_az_bins)} bins over {_bin_width * int(n_az_bins):.1f}°)")

        with st.form("cake_background_form"):
            bg_cols = st.columns(4)
            with bg_cols[0]:
                bg_smoothing_sigma = st.number_input(
                    "Smoothing σ", value=10.0, min_value=0.0, step=1.0,
                    help="Width (in 2θ points) of the Gaussian smoothing applied before "
                         "peak detection. Larger σ suppresses noise so only broader "
                         "features are flagged as peaks; too large can merge or miss "
                         "sharp peaks. Affects peak finding only — not the fitted curve.")
                bg_poly_degree = st.number_input(
                    "Polynomial degree", value=45, min_value=1, step=1,
                    help="Order of the Chebyshev polynomial fitted to the background "
                         "points of each binned profile. Higher follows more curvature "
                         "(undulating background) but can start bending into peak tails; "
                         "lower is stiffer and smoother.")
            with bg_cols[1]:
                bg_prominence_factor = st.number_input(
                    "Prominence factor", value=0.25, min_value=0.0, step=0.01, format="%.3f",
                    help="Peak-detection sensitivity, used by BOTH stages below. A feature "
                         "must rise at least this fraction of the current maximum to be "
                         "flagged and excluded. Lower → more sensitive (excludes weaker "
                         "peaks); set > 1 to select no peaks at all.")
                bg_peak_iterations = st.number_input(
                    "Peak-search iterations", value=1, min_value=0, step=1,
                    help="Stage 1 (before fitting). One peak-search pass on the smoothed "
                         "profile is always done; this is the number of ADDITIONAL passes. "
                         "Each extra pass excludes the peaks found so far, lowering the "
                         "running maximum so progressively weaker peaks get caught, before "
                         "the initial background is fitted. 0 = the single baseline pass.")
                bg_iterations = st.number_input(
                    "Refinement iterations", value=1, min_value=0, step=1,
                    help="Stage 2 (after fitting). Residual-refinement passes: the fitted "
                         "background is subtracted and the residual is searched for peaks "
                         "the pre-fit stage missed (shallow peaks on the background); those "
                         "are excluded and the background is refitted. Stops early once "
                         "none are found. 0 = no residual refinement.")
            with bg_cols[2]:
                bg_exclusion_window = st.number_input(
                    "Peak exclusion window", value=25, min_value=0, step=1,
                    help="Number of 2θ points removed on EACH side of every detected "
                         "peak before fitting, so peak flanks don't pull the background "
                         "up. Increase if peak wings drag the fit upward; decrease if too "
                         "much genuine background is being discarded.")
            with bg_cols[3]:
                bg_negative_clip = st.number_input(
                    "Negative clip", value=-10.0, step=1.0,
                    help="After subtracting the background, any value below this is set "
                         "to 0. Removes large negative dips left where the fit overshoots "
                         "data gaps. More negative keeps more of the (noisier) sub-zero "
                         "signal; nearer 0 forces a cleaner but harder floor.")
                bg_gap_fill = st.checkbox(
                    "Gap fill (interpolate)", value=True,
                    help="Bridge large detector gaps by inserting pseudo background points "
                         "across them, linearly interpolated from the real background on "
                         "either side, so the polynomial can't swing wildly through empty "
                         "regions.")
                bg_gap_min_width = st.number_input(
                    "Gap min width (points)", value=5, min_value=1, step=1,
                    help="Minimum width (in 2θ points) of a contiguous run of zeros to be "
                         "treated as a gap and bridged with pseudo points. Smaller catches "
                         "more/narrower gaps.")
                bg_gap_pad = st.number_input(
                    "Gap edge pad (points)", value=10, min_value=0, step=1,
                    help="Extra points added on each side of a gap before interpolating the "
                         "pseudo points. Pushes the interpolation anchors past the intensity "
                         "taper at the gap edges so the pseudo points sit at the true "
                         "baseline, not the (weaker) tapered values. Increase if pseudo "
                         "points look too low around gaps.")
            bg_submitted = st.form_submit_button("Compute background")

        if bg_submitted:
            # Fit kwargs, stored so the lineout inspector reproduces the same fit/peaks.
            _bg_fit_kwargs = dict(
                poly_degree=int(bg_poly_degree),
                smoothing_sigma=float(bg_smoothing_sigma),
                prominence_factor=float(bg_prominence_factor),
                peak_iterations=int(bg_peak_iterations),
                iterations=int(bg_iterations),
                exclusion_window=int(bg_exclusion_window),
                gap_fill=bool(bg_gap_fill),
                gap_min_width=int(bg_gap_min_width),
                gap_pad=int(bg_gap_pad),
            )
            with st.spinner("Fitting per-bin background..."):
                st.session_state.cake_background = cp.compute_cake_background(
                    _cake,
                    n_bins=int(n_az_bins),
                    negative_clip=float(bg_negative_clip),
                    **_bg_fit_kwargs,
                )
            st.session_state.cake_background_params = _bg_fit_kwargs
            st.session_state.cake_background_nbins = int(n_az_bins)
            # Bump the version so any prepared downloads are treated as outdated.
            st.session_state.cake_background_version = \
                st.session_state.get("cake_background_version", 0) + 1

        # Alternative to fitting: load a pre-made background and subtract it directly.
        with st.expander("Load a pre-made background instead of fitting"):
            _loaded_bg_file = st.file_uploader(
                "Background file (.txt or .tiff, matching this cake's size)",
                type=["txt", "tif", "tiff"], key="cake_bg_upload")
            _loaded_neg_clip = st.number_input(
                "Negative clip (for loaded background)", value=-10.0, step=1.0,
                key="cake_loaded_negclip")
            if _loaded_bg_file is not None and st.button("Load & subtract this background"):
                try:
                    _grid = cp.load_grid_file(_loaded_bg_file, _loaded_bg_file.name)
                except Exception as e:
                    st.error(f"Could not read background file: {e}")
                else:
                    if _grid.shape != _cake.intensity.shape:
                        st.error(f"Background shape {_grid.shape} does not match the cake "
                                 f"{_cake.intensity.shape}.")
                    else:
                        st.session_state.cake_background = cp.background_from_grid(
                            _cake, _grid, negative_clip=float(_loaded_neg_clip))
                        st.session_state.cake_background_params = {}
                        st.session_state.cake_background_nbins = int(n_az_bins)
                        st.session_state.cake_background_version = \
                            st.session_state.get("cake_background_version", 0) + 1
                        st.success("Loaded background applied.")

        _bg = st.session_state.get("cake_background")
        # Guard against a stale result from a previously-loaded cake of a different size.
        if _bg is not None and _bg.background.shape == _cake.intensity.shape:
            # Bin assignment for the lineout uses the n_bins the background was computed
            # with (so the displayed bin matches the fit even if the input changed since).
            _nbins_used = int(st.session_state.get("cake_background_nbins", n_az_bins))
            _edges, _bin_index, _bin_width = cp.assign_azimuth_bins(_cake.azimuth, _nbins_used)

            # Azimuth selector: slide to pick a bin; the choice is shown as a translucent
            # band on both heatmaps below and drives the lineout inspector.
            _az_min, _az_max = float(_cake.azimuth.min()), float(_cake.azimuth.max())
            _default_az = float(min(max(
                st.session_state.get("cake_lineout_azimuth", float(_cake.azimuth[_n_az // 2])),
                _az_min), _az_max))
            _sel_az = st.slider(
                "Azimuth for lineout (°)", min_value=_az_min, max_value=_az_max,
                value=_default_az, step=float(_bin_width),
                help="Selects the azimuth bin shown highlighted on the cakes and in the "
                     "lineout below.")
            st.session_state.cake_lineout_azimuth = float(_sel_az)
            _sel_bin = int(_bin_index[int(np.argmin(np.abs(_cake.azimuth - _sel_az)))])
            _rows_in_bin = np.where(_bin_index == _sel_bin)[0]
            _bin_lo = float(_cake.azimuth[_rows_in_bin].min())
            _bin_hi = float(_cake.azimuth[_rows_in_bin].max())
            _band = (float(_edges[_sel_bin]), float(_edges[_sel_bin + 1]))

            bg_result_cols = st.columns(2)
            with bg_result_cols[0]:
                st.plotly_chart(
                    cp.plot_grid_heatmap(_cake, _bg.background, "Fitted background",
                                         percentile=cake_percentile, highlight_band=_band),
                    width='stretch',
                )
            with bg_result_cols[1]:
                st.plotly_chart(
                    cp.plot_grid_heatmap(_cake, _bg.subtracted, "Background-subtracted",
                                         percentile=cake_percentile, highlight_band=_band),
                    width='stretch',
                )

            # Lineout inspector: raw / fitted background / pseudo points / subtracted,
            # averaged over the selected azimuth bin. Traces toggle via the legend.
            st.markdown(
                f"**Lineout inspector** — azimuth bin {_sel_bin} "
                f"({_bin_lo:.1f}° to {_bin_hi:.1f}°, {_rows_in_bin.size} rows). "
                f"Toggle traces via the legend.")
            st.plotly_chart(
                cp.plot_azimuth_lineout(
                    _cake, _bg, _rows_in_bin,
                    sample_kwargs=st.session_state.get("cake_background_params", {}),
                ),
                width='stretch',
            )

            # Download / export. Files are generated for the CURRENT background only and
            # invalidated whenever it is recomputed/reloaded, so an outdated result can
            # never be downloaded. Generation is behind a button to avoid rebuilding the
            # (large) files on every rerun while tuning.
            st.markdown("**Download** — Dioptas-format `.txt` (re-loadable) or 32-bit float `.tiff`")
            _bg_version = st.session_state.get("cake_background_version", 0)
            _dl_data = st.session_state.get("cake_download_data")
            _dl_current = _dl_data is not None and _dl_data.get("version") == _bg_version
            if not _dl_current:
                if st.button("Prepare download files"):
                    with st.spinner("Preparing export files..."):
                        st.session_state.cake_download_data = {
                            "version": _bg_version,
                            "bg_txt": cp.cake_grid_to_txt_bytes(
                                _cake.twotheta, _cake.azimuth, _bg.background),
                            "bg_tiff": cp.grid_to_tiff_bytes(_bg.background),
                            "sub_txt": cp.cake_grid_to_txt_bytes(
                                _cake.twotheta, _cake.azimuth, _bg.subtracted),
                            "sub_tiff": cp.grid_to_tiff_bytes(_bg.subtracted),
                        }
                    _dl_data = st.session_state.cake_download_data
                    _dl_current = True
                else:
                    st.caption("Press **Prepare download files** to generate downloads for "
                               "the current background subtraction.")
            if _dl_current:
                _dl = st.columns(4)
                _dl[0].download_button("Background (.txt)", _dl_data["bg_txt"],
                                       file_name="cake_background.txt", mime="text/plain")
                _dl[1].download_button("Background (.tiff)", _dl_data["bg_tiff"],
                                       file_name="cake_background.tiff", mime="image/tiff")
                _dl[2].download_button("Subtracted (.txt)", _dl_data["sub_txt"],
                                       file_name="cake_subtracted.txt", mime="text/plain")
                _dl[3].download_button("Subtracted (.tiff)", _dl_data["sub_tiff"],
                                       file_name="cake_subtracted.tiff", mime="image/tiff")

    # --- 2D Refinement Tools: experimental peak extraction from the subtracted cake ---
with tab_peaks:
    # --- 2D Refinement Tools: experimental peak extraction ---
    # Works on ANY background-subtracted cake: the Background Subtraction result from the
    # Cake tab, or a pre-subtracted Dioptas .txt uploaded here directly.
    st.subheader("2D Refinement Tools")
    st.markdown("**Peak Extraction** — extract (azimuth, 2θ) points per hkl ring from a "
                "background-subtracted cake, grouped by reflection.")
    with st.expander("ℹ️ How the peak search works"):
        st.markdown(cp.PEAK_EXTRACTION_HELP_MD)

    _have_bg = (
        st.session_state.get("cake_background") is not None
        and st.session_state.get("imported_cake") is not None
        and st.session_state.cake_background.subtracted.shape
            == st.session_state.imported_cake.intensity.shape)
    _src_opts = (["Background-subtraction result", "Upload subtracted cake (.txt)"]
                 if _have_bg else ["Upload subtracted cake (.txt)"])
    _src = st.radio("Subtracted image source", _src_opts, horizontal=True,
                    key="cake_extract_source")

    _xc = _grid = None
    if _src == "Background-subtraction result" and _have_bg:
        _xc = st.session_state.imported_cake
        _grid = st.session_state.cake_background.subtracted
    else:
        _sub_file = st.file_uploader(
            "Background-subtracted cake (Dioptas .txt)", type=["txt"],
            key="cake_subtracted_upload",
            help="A cake that is already background-subtracted (exported here, or from "
                 "Dioptas). The .txt carries the 2θ/azimuth axes the peak search needs.")
        if _sub_file is not None:
            try:
                _xc = cp.load_cake_data(_sub_file, filename=_sub_file.name)
                _grid = _xc.intensity
            except ValueError as e:
                st.error(f"Failed to load subtracted cake: {e}")

    if _xc is None or _grid is None:
        st.info("Provide a subtracted image to search: compute or load a background in "
                "the Cake tab, or upload a pre-subtracted Dioptas .txt here.")
    else:
        # Make the resolved subtracted image available to the Refinement tab (Stage 3).
        st.session_state.subtracted_image = (_xc, _grid)
        # Show the subtracted image so the user can inspect the data before setting params.
        st.plotly_chart(
            cp.plot_grid_heatmap(_xc, _grid, "Subtracted cake to search",
                                 percentile=cake_percentile),
            width='stretch')
        _tth_lo, _tth_hi = float(_xc.twotheta.min()), float(_xc.twotheta.max())
        _tth_step = float(np.median(np.diff(_xc.twotheta))) if _xc.twotheta.size > 1 else 0.01
        _tth_step = abs(_tth_step) or 0.01
        _def_bins = int(st.session_state.get("cake_background_nbins",
                                             max(1, _xc.azimuth.size // 2)))
        _def_bins = min(_def_bins, int(_xc.azimuth.size))
        with st.form("cake_extract_form"):
            ex = st.columns(4)
            with ex[0]:
                ex_tmin = st.number_input("2θ min (°)", min_value=_tth_lo,
                    max_value=_tth_hi, value=_tth_lo, step=0.5,
                    help="Lower edge of the 2θ window searched. Narrow it to the rings "
                         "of interest to avoid seeding noise or unwanted phases.")
                ex_tmax = st.number_input("2θ max (°)", min_value=_tth_lo,
                    max_value=_tth_hi, value=_tth_hi, step=0.5,
                    help="Upper edge of the 2θ window searched.")
            with ex[1]:
                ex_maxpk = st.number_input("Max hkl peaks (groups)", min_value=1,
                    value=6, step=1,
                    help="How many rings/hkl groups to seed (the strongest N in the "
                         "window). Set to the number of reflections you expect. "
                         "Typical 3–12.")
                ex_shape = st.selectbox("Peak shape", ["PseudoVoigt", "Gaussian"],
                    help="Profile fitted to each peak for its refined 2θ. Pseudo-Voigt "
                         "suits most powder peaks; Gaussian for clean symmetric ones.")
            with ex[2]:
                ex_bins = st.number_input("Azimuth bins", min_value=1,
                    max_value=int(_xc.azimuth.size), value=_def_bins, step=1,
                    help="Bins the cake is averaged into for the search. More bins = "
                         "finer azimuthal detail but noisier lineouts. Max = one bin per "
                         "azimuth row ({}).".format(int(_xc.azimuth.size)))
                ex_seedp = st.number_input("Seed sensitivity", min_value=0.001,
                    max_value=1.0, value=0.03, step=0.005, format="%.3f",
                    help="Fraction of the azimuth max-projection's maximum a ring must "
                         "exceed to seed a group. Lower seeds fainter rings. "
                         "Typical 0.01–0.1.")
                ex_seedgap = st.number_input("Min seed spacing (°)", min_value=0.0,
                    value=0.5, step=0.1, format="%.2f",
                    help="Smallest 2θ gap allowed between two seeds, so one broad ring "
                         "is not seeded twice. Set just below your closest real ring "
                         "spacing. Typical 0.3–1.0°.")
            with ex[3]:
                ex_dets = st.number_input("Detection σ", min_value=0.0, value=5.0,
                    step=0.5,
                    help="A ring is recorded in a bin only where its peak rises this many "
                         "robust σ above the window's noise. Lower catches fainter arcs "
                         "(more noise); higher keeps only strong segments. Typical 3–8.")
                ex_mshift = st.number_input("Ring-track tol (°, 0=auto)", min_value=0.0,
                    value=0.0, step=0.1,
                    help="How far (in 2θ) a peak may sit from a ring's running centre to "
                         "still join it. Larger tolerates bigger strain shifts but risks "
                         "jumping to a neighbour. Typical 0.2–1.0°; 0 = auto (~0.4x seed "
                         "spacing).")
                ex_fitwin = st.number_input("Fit window (samples)", min_value=3,
                    value=15, step=1,
                    help="Half-width, in 2θ samples, of the local slice fitted around each "
                         "detected peak (the fit spans ±this many points, 2N+1 total). "
                         "Cover the full peak width without reaching neighbouring rings. "
                         "Typical 8–25. Data 2θ step ≈ {:.4f}°/sample.".format(_tth_step))
                st.caption("Fit window ≈ ±{:.3f}° around each peak "
                           "(±{} samples).".format(int(ex_fitwin) * _tth_step,
                                                   int(ex_fitwin)))
            ex_submit = st.form_submit_button("Extract peaks")

        if ex_submit:
            _ms = None if ex_mshift <= 0 else float(ex_mshift)
            _seed_pts = max(1, int(round(float(ex_seedgap) / _tth_step)))
            with st.spinner("Extracting peaks..."):
                _pk, _seeds = cp.extract_and_group_peaks(
                    _xc, _grid, tth_min=float(ex_tmin), tth_max=float(ex_tmax),
                    n_bins=int(ex_bins), max_peaks=int(ex_maxpk), peak_shape=ex_shape,
                    seed_prominence=float(ex_seedp), min_seed_distance=_seed_pts,
                    detect_sigma=float(ex_dets), fit_window=int(ex_fitwin), max_shift=_ms)
            st.session_state.extracted_peaks = _pk
            # Store the params the peak-fit lineout viewer needs to reproduce the binning.
            st.session_state.extracted_peaks_params = {
                "tth_min": float(ex_tmin), "tth_max": float(ex_tmax),
                "n_bins": int(ex_bins), "peak_shape": ex_shape}
            st.session_state.extracted_peaks_ver = \
                st.session_state.get("extracted_peaks_ver", 0) + 1
            if _pk.empty:
                st.warning("No peaks found — widen the 2θ range, lower Detection σ, or "
                           "lower Seed sensitivity.")
            else:
                st.success(f"Extracted {len(_pk)} points in "
                           f"{int(_pk['group'].nunique())} groups "
                           f"(seeds at 2θ = {', '.join(f'{s:.2f}' for s in _seeds)}°).")

        _pk = st.session_state.get("extracted_peaks")
        if _pk is not None and not _pk.empty:
            st.caption("Correct assignments below: edit the **group** column (hkl index) "
                       "or delete outlier rows — the plot updates live.")
            _edited = st.data_editor(
                _pk, key=f"cake_peaks_editor_{st.session_state.get('extracted_peaks_ver', 0)}",
                num_rows="dynamic", use_container_width=True,
                column_config={
                    "bin": st.column_config.NumberColumn("bin", disabled=True),
                    "azimuth": st.column_config.NumberColumn("azimuth (°)", disabled=True, format="%.1f"),
                    "2th": st.column_config.NumberColumn("2θ (°)", disabled=True, format="%.3f"),
                    "intensity": st.column_config.NumberColumn("intensity", disabled=True, format="%.0f"),
                    "fwhm": st.column_config.NumberColumn("fwhm (°)", disabled=True, format="%.3f"),
                    "gl": st.column_config.NumberColumn("gl (G↔L)", disabled=True, format="%.2f",
                        help="Fitted Gaussian↔Lorentzian fraction (0 = Gaussian, 1 = "
                             "Lorentzian); Pseudo-Voigt only."),
                    "group": st.column_config.NumberColumn("group (hkl)", min_value=-1, step=1),
                })
            # Dynamic-row edits can introduce NaN group cells (added/blanked rows);
            # coerce to a safe int so the int(group) calls below never raise.
            if "group" in _edited.columns:
                _edited["group"] = _edited["group"].fillna(-1).astype(int)
            # --- Assign hkl reflections to the groups ---
            st.markdown("**Assign hkl reflections** — label each group with its hkl "
                        "(and optional material/phase). The mean 2θ identifies the ring.")
            _summary = cp.summarise_groups(_edited)
            _saved = st.session_state.setdefault("cake_group_labels", {})
            # Rebuild the label table only when the SET of groups changes, so edits within
            # a fixed group set are not overwritten each rerun (avoids data_editor double-apply).
            _grp_key = f"{st.session_state.get('extracted_peaks_ver', 0)}_" + \
                "_".join(str(int(g)) for g in _summary["group"])
            if st.session_state.get("cake_label_key") != _grp_key:
                _rows = [{"group": int(_r["group"]), "points": int(_r["points"]),
                          "mean 2θ": round(float(_r["mean_2th"]), 3),
                          "hkl": _saved.get(int(_r["group"]), {}).get("hkl", ""),
                          "material": _saved.get(int(_r["group"]), {}).get("material", "")}
                         for _, _r in _summary.iterrows()]
                st.session_state.cake_label_base = pd.DataFrame(
                    _rows, columns=["group", "points", "mean 2θ", "hkl", "material"])
                st.session_state.cake_label_key = _grp_key
            _labels_edited = st.data_editor(
                st.session_state.cake_label_base, key=f"cake_label_editor_{_grp_key}",
                use_container_width=True, hide_index=True,
                column_config={
                    "group": st.column_config.NumberColumn("group", disabled=True),
                    "points": st.column_config.NumberColumn("points", disabled=True),
                    "mean 2θ": st.column_config.NumberColumn("mean 2θ (°)", disabled=True, format="%.3f"),
                    "hkl": st.column_config.TextColumn("hkl", help="e.g. 111, 200, 220"),
                    "material": st.column_config.TextColumn("material", help="optional phase / material name"),
                })
            # Persist labels by group and apply them to the peaks.
            _glabels = {}
            for _, _r in _labels_edited.iterrows():
                _hkl, _mat = str(_r["hkl"]).strip(), str(_r["material"]).strip()
                _saved[int(_r["group"])] = {"hkl": _hkl, "material": _mat}
                _glabels[int(_r["group"])] = _hkl if not _mat else f"{_hkl} ({_mat})"
            _labeled = _edited.copy()
            _labeled["hkl"] = _labeled["group"].map(lambda g: _saved.get(int(g), {}).get("hkl", ""))
            _labeled["material"] = _labeled["group"].map(lambda g: _saved.get(int(g), {}).get("material", ""))
            st.session_state.extracted_peaks_final = _labeled

            st.download_button(
                "Download labelled peaks (.csv)",
                _labeled.to_csv(index=False).encode("utf-8"),
                file_name="extracted_peaks_labelled.csv", mime="text/csv")
            st.plotly_chart(
                cp.plot_extracted_peaks(_xc, _grid, _labeled, percentile=cake_percentile,
                                        group_labels=_glabels),
                width='stretch')

            # --- Lineout peak-fit viewer (per azimuth bin) ---
            _pf = st.session_state.get("extracted_peaks_params", {})
            _vb_nbins = max(1, min(int(_pf.get("n_bins", max(1, _xc.azimuth.size // 2))),
                                   int(_xc.azimuth.size)))
            _vb_tmin = float(_pf.get("tth_min", float(_xc.twotheta.min())))
            _vb_tmax = float(_pf.get("tth_max", float(_xc.twotheta.max())))
            _edges_v, _bidx_v, _bw_v = cp.assign_azimuth_bins(_xc.azimuth, _vb_nbins)
            _azmin_v, _azmax_v = float(_xc.azimuth.min()), float(_xc.azimuth.max())
            _def_v = float(min(max(
                st.session_state.get("peakfit_azimuth", float(_xc.azimuth[_xc.azimuth.size // 2])),
                _azmin_v), _azmax_v))
            st.markdown("**Lineout peak-fit viewer** — the binned 1D profile with every "
                        "fitted peak overlaid, for the azimuth bin you pick. Toggle traces "
                        "via the legend.")
            _sel_v = st.slider(
                "Azimuth for peak-fit lineout (°)", min_value=_azmin_v, max_value=_azmax_v,
                value=_def_v, step=float(_bw_v),
                help="Selects the azimuth bin whose 1D lineout and peak fits are shown below "
                     "(same binning as the extraction).")
            st.session_state.peakfit_azimuth = float(_sel_v)
            _sel_bin_v = int(_bidx_v[int(np.argmin(np.abs(_xc.azimuth - _sel_v)))])
            _n_in_bin = int((_labeled["bin"] == _sel_bin_v).sum()) \
                if "bin" in _labeled.columns else 0
            st.caption(f"Bin {_sel_bin_v}: {_n_in_bin} extracted peak(s) in this azimuth bin.")
            st.plotly_chart(
                cp.plot_bin_peak_fits(_xc, _grid, _labeled, bin_index=_sel_bin_v,
                                      n_bins=_vb_nbins, tth_min=_vb_tmin, tth_max=_vb_tmax,
                                      group_labels=_glabels),
                width='stretch')

with tab_sim:
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.subheader("Upload Files")
        uploaded_file = st.file_uploader("Elastic and hkl csv", type=["csv"])

    if uploaded_file is not None:
        with col2:
            st.subheader("")
            poni_file = st.file_uploader("Poni", type=["poni"])
        with col3:
            st.subheader("")
            batch_upload = st.file_uploader("Batch XRD file", type=["csv"])
        with col4:
            st.subheader("")
            twoD_XRD = st.file_uploader("2D XRD tiff", type=["tiff"])

        #Define download_data if not initialised
        if "download_data" not in st.session_state:
            st.session_state.download_data = {}
        #Initialise the "previous" download format to track changes. Defaults to "Excel (.xlsx)"
        if "prev_download_format" not in st.session_state:
            st.session_state.prev_download_format = st.session_state.get("download_format", "Excel (.xlsx)")

        #Section for downloading computed data
        columns = st.columns(6)
        with columns[0]:
            st.subheader("Download Data")
            with st.form("download_form"):
                    st.selectbox(
                        "Set download format",
                        ["Excel (.xlsx)", "OpenDocument (.ods)", "ZIP of CSVs (.zip)"],
                        index=0,
                        key = "download_format"
                    )
                    submitted = st.form_submit_button("Set format")
            st.write(st.session_state.download_format)
        
            #Reformat the data only if selection changed
            if submitted:
                if st.session_state.download_format != st.session_state.prev_download_format:
                    if st.session_state.download_data:
                        st.write("Format changed → reprocessing data")
                    #Reformat the available data accordingly
                    for key,data in st.session_state.download_data.items():
                        if key in ["epsilon_psi", "cake"]:
                            datasource = data["datasource"]
                            if st.session_state.download_format == "Excel (.xlsx)":
                                output_buffer = io.BytesIO()
                
                                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                                    for hkl_label, df in datasource.items():
                                        sheet_name = f"hkl_{hkl_label}"
                                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                                        worksheet = writer.sheets[sheet_name]
                                        for i, col in enumerate(df.columns):
                                            max_width = max(
                                                df[col].astype(str).map(len).max(),
                                                len(col)
                                            ) + 2
                                            worksheet.set_column(i, i, max_width)
                
                                output_buffer.seek(0)
                                buffer = output_buffer
                                file = data["filename"].split(".")
                                filename = "{}.xlsx".format(file[0])
                                mime =("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                                store_download(key, datasource, buffer, filename, mime)
                
                            elif st.session_state.download_format == "OpenDocument (.ods)":
                                output_buffer = io.BytesIO()
                
                                with pd.ExcelWriter(output_buffer, engine='odf') as writer:
                                    for hkl_label, df in datasource.items():
                                        df.to_excel(writer, sheet_name=f"hkl_{hkl_label}", index=False)
                
                                output_buffer.seek(0)
                                buffer = output_buffer
                                file = data["filename"].split(".")
                                filename = "{}.ods".format(file[0])
                                mime =("application/vnd.oasis.opendocument.spreadsheet")
                                store_download(key, datasource, buffer, filename, mime)

                            elif st.session_state.download_format == "ZIP of CSVs (.zip)":
                                output_buffer = io.BytesIO()
        
                                with zipfile.ZipFile(output_buffer, "w") as zf:
                                    for hkl_label, df in datasource.items():
                                        csv_buffer = io.StringIO()
                                        df.to_csv(csv_buffer, index=False)
                                        zf.writestr(f"{hkl_label}.csv", csv_buffer.getvalue())
                            
                                output_buffer.seek(0)
                                buffer = output_buffer
                                file = data["filename"].split(".")
                                filename = "{}.zip".format(file[0])
                                mime =("application/zip")
                                store_download(key, datasource, buffer, filename, mime)
                            else:
                                pass
                    # update stored value
                    st.session_state.prev_download_format = st.session_state.download_format
                else:
                    pass
            
        if st.session_state.download_data:
            columns = st.columns(12)
            download_data = st.session_state.download_data
            items = list(download_data.items())
            for i, (key, data) in enumerate(items):
                with columns[i]:
                    # Persistent download buttons
                    if st.download_button(
                        label=f"📥 Download {data["filename"]}",
                        data=data["buffer"],
                        file_name=data["filename"],
                        mime=data["mime"],
                        key=f"download_{i}"  # unique key required
                    ):
                        # Auto-clear
                        st.session_state.download_data.pop(key, None)
            
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2,3,1,2,1,1,1,1])
        with col1:
            st.subheader("Reflections/Intensities")
        with col2:
            st.subheader("Material")
        with col3:
            st.subheader("Elastic")
        with col4:
            st.subheader("Stress")
        with col5:
            st.subheader("Computation")
        with col6:
            st.subheader("Preferred Orientation")

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns([2,1,1,1,1,1,1,1,1,1,1])

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        file_obj = st.session_state.get("uploaded_file", None)
        # --- Read and split file ---
        content = file_obj.getvalue().decode("utf-8")
        lines = content.strip().splitlines()
        # --- Separate metadata and data lines ---
        metadata = {}
        data_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Extract metadata lines of form "# key: value"
                if ':' in line:
                    key, val = line[1:].split(':', 1)
                    try:
                        metadata[key.strip()] = float(val)
                    except:
                        metadata[key.strip()] = val.strip()
            else:
                data_lines.append(line)

        symmetry = metadata["symmetry"]
        #Check the correct data has been included for the respective symmetry
        if symmetry == "cubic":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C12','C44','sig11','sig22','sig33','chi',}
        elif symmetry == "hexagonal":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C44','sig11','sig22','sig33','chi'}
        elif symmetry == "tetragonal_A":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C44','C66','sig11','sig22','sig33','chi'}
        elif symmetry == "tetragonal_B":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13','C16','C44','C66','sig11','sig22','sig33','chi'}
        elif symmetry == "orthorhombic":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C22','C33','C12','C13','C23','C44','C55','C66','sig11','sig22','sig33','chi'}
        elif symmetry == "trigonal_A":
            required_keys = {'a','b','c','alpha','beta','gamma','wavelength','C11','C33','C12','C13', 'C14','C44','sig11','sig22','sig33','chi'}
        else:
            st.error("{} symmetry is not yet supported".format(symmetry))
            required_keys = {}
    
        #Optional for off-diagonal stress terms - otherwise default to zero
        optional_keys = {'sig12', 'sig13', 'sig23'}

        metadata_keys = set(metadata) - {'symmetry'} #Drops the symmetry key from the check as it is assumed present

        missing_keys = required_keys - metadata_keys
        allowed_keys = required_keys | optional_keys
        extra_keys = metadata_keys - allowed_keys
    
        if missing_keys:
            st.error(f"Missing required keys: {', '.join(missing_keys)}")
            st.write(f"CSV must contain at least: {', '.join(required_keys)}")
            st.stop()
    
        if extra_keys:
            st.warning(f"Unexpected keys found: {', '.join(extra_keys)}")

        #Set sig12, 13, 23 if not provided in the input file
        for key in ['sig12', 'sig13', 'sig23']:
            if key in metadata.keys():
                pass
            else:
                metadata[key] = 0.0 #Set default value to zero
    
        # --- Parse HKL + intensity section ---
        try:
            hkl_df = pd.read_csv(io.StringIO("\n".join(data_lines)))
        except Exception as e:
            st.error(f"Error reading HKL section: {e}")
            st.stop()
        # Validate required columns
        required_cols = {'h', 'k', 'l', 'intensity'}
        if not required_cols.issubset(hkl_df.columns):
            st.error(f"HKL section must have columns: {', '.join(required_cols)}")
            st.stop()
        else:
            # Ensure numeric conversion
            hkl_df[['h', 'k', 'l']] = hkl_df[['h', 'k', 'l']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            hkl_df['intensity'] = pd.to_numeric(hkl_df['intensity'], errors='coerce').fillna(1.0)
            hkl_list = hkl_df[['h', 'k', 'l']].drop_duplicates().values.tolist()
        
            #Initialise lists/dictionaries
            selected_hkls = []
            intensities = []
            selected_indices = []
            peak_intensity_default = {}

            if "params" not in st.session_state:
                st.session_state.params = {
                    "a_val": float(metadata['a']),
                    "b_val": float(metadata['b']),
                    "c_val": float(metadata['c']),
                    "alpha": float(metadata['alpha']),
                    "beta": float(metadata['beta']),
                    "gamma": float(metadata['gamma']),
                    "chi": float(metadata['chi']),
                    "wavelength": float(metadata['wavelength']),
                    **{k.lower(): metadata[k] for k in metadata.keys() if k.startswith("C")},
                    "sigma_11": float(metadata["sig11"]),
                    "sigma_22": float(metadata["sig22"]),
                    "sigma_33": float(metadata["sig33"]),
                    "sigma_12": float(metadata["sig12"]),
                    "sigma_13": float(metadata["sig13"]),
                    "sigma_23": float(metadata["sig23"])
                }
            with col1:
                for i, hkl in enumerate(hkl_list):
                        # Find matching row to get intensity
                        h_match = (hkl_df['h'] == hkl[0]) & (hkl_df['k'] == hkl[1]) & (hkl_df['l'] == hkl[2])
                        default_intensity = float(hkl_df[h_match]['intensity'].values[0]) if h_match.any() else 1.0
                        peak_intensity_default[f"intensity_{i}"] = default_intensity
                
                # Initialize state for peak intensity
                if "intensities" not in st.session_state:
                    st.session_state.intensities = peak_intensity_default.copy()

                for i, hkl in enumerate(hkl_list):
                    cols = st.columns(2)    
                    with cols[0]:
                        label = f"hkl = ({int(hkl[0])}, {int(hkl[1])}, {int(hkl[2])})"
                        selected = st.checkbox(label, value=True, key=f"chk_{i}")
                    with cols[1]:
                        st.session_state.intensities[f"intensity_{i}"] = st.number_input(
                            f"Intensity_{i}",
                            min_value=0.0,
                            value=st.session_state.intensities[f"intensity_{i}"],
                            step=1.0,
                            label_visibility="collapsed"
                        )

                    if selected:
                        selected_hkls.append(hkl)
                        selected_indices.append(i)  # Save which index was selected
                        intensities.append(st.session_state.intensities[f"intensity_{i}"])

                #One multiplier applied on top of every intensity above, so the overall
                #level can be set with a single value. Refineable in the panel below; a
                #refinement writes the fitted value straight back into this box.
                st.session_state.intensity_global_multiplier = st.number_input(
                    "Intensity global multiplier",
                    min_value=0.0,
                    value=float(st.session_state.get("intensity_global_multiplier", 1.0)),
                    step=0.1,
                    format="%.4f",
                    help="Scales the whole simulated pattern. 1.0 leaves it unscaled. "
                         "Applied to the 1D-XRD and Overlay XRD plots and to the "
                         "refinement.")

            with col2:
                symmetry_options = ["cubic", "hexagonal", "tetragonal_A", "tetragonal_B", "orthorhombic", "trigonal_A"]
                if metadata['symmetry'] in symmetry_options:
                    default_index = symmetry_options.index(metadata['symmetry'])
                else:
                    default_index = 0  # fallback
                    st.write("{} symmetry not supported. Choose from the options below". format(metadata['symmetry']))
                symmetry = st.selectbox("Symmetry:",symmetry_options, index=default_index)
                st.session_state.params["wavelength"] = st.number_input("Wavelength (Å)", value=st.session_state.params["wavelength"], step=0.01, format="%.4f")
                st.session_state.params["chi"] = st.number_input("Chi angle (deg)", value=st.session_state.params["chi"], step=0.01, format="%.3f")            
            with col3:
                st.session_state.params["a_val"] = st.number_input("Lattice a (Å)", value=st.session_state.params["a_val"], step=0.01, format="%.4f")
                st.session_state.params["b_val"] = st.number_input("Lattice b (Å)", value=st.session_state.params["b_val"], step=0.01, format="%.4f")
                st.session_state.params["c_val"] = st.number_input("Lattice c (Å)", value=st.session_state.params["c_val"], step=0.01, format="%.4f")
            with col4:
                st.session_state.params["alpha"] = st.number_input("alpha (deg)", value=st.session_state.params["alpha"], step=0.1, format="%.3f")
                st.session_state.params["beta"] = st.number_input("beta (deg)", value=st.session_state.params["beta"], step=0.1, format="%.3f")
                st.session_state.params["gamma"] = st.number_input("gamma (deg)", value=st.session_state.params["gamma"], step=0.1, format="%.3f")
            with col5:
                # Dynamically build the list of Cij keys present in params
                c_keys = [key for key in st.session_state.params.keys() if key.startswith('c') and key not in ["c_val", "chi"]]
                cijs = {}
                for key in c_keys:
                    st.session_state.params[key] = st.number_input(key, value=st.session_state.params[key])
                    cijs[key] = st.session_state.params.get(key)
            with col6:
                st.session_state.params["sigma_11"] = st.number_input("σ₁₁", value=st.session_state.params["sigma_11"], step=0.1, format="%.3f")
                st.session_state.params["sigma_22"] = st.number_input("σ₂₂", value=st.session_state.params["sigma_22"], step=0.1, format="%.3f")
                st.session_state.params["sigma_33"] = st.number_input("σ₃₃", value=st.session_state.params["sigma_33"], step=0.1, format="%.3f")
                st.markdown("t: {}".format(round(st.session_state.params["sigma_33"] - st.session_state.params["sigma_11"],3)))
            with col7:
                st.session_state.params["sigma_12"] = st.number_input("σ₁₂", value=st.session_state.params["sigma_12"], step=0.1, format="%.3f")
                st.session_state.params["sigma_13"] = st.number_input("σ₁₃", value=st.session_state.params["sigma_13"], step=0.1, format="%.3f")
                st.session_state.params["sigma_23"] = st.number_input("σ₂₃", value=st.session_state.params["sigma_23"], step=0.1, format="%.3f")
            with col8:
                Funamori_broadening = st.checkbox("Include broadening", value=True)
                # Independent ε–ψ sampling. psi = output (plot) axis; phi, alpha = integration axes.
                psi_steps = st.number_input("ψ steps (output/plot)",  value=60, min_value=2, step=1)
                phi_steps = st.number_input("φ steps (integration)",  value=18, min_value=1, step=1)
                alpha_steps = st.number_input("α steps (integration)",  value=18, min_value=1, step=1)
                st.caption("Total points: {:,}  ({}ψ × {}φ × {}α)".format(
                    int(psi_steps) * int(phi_steps) * int(alpha_steps),
                    int(psi_steps), int(phi_steps), int(alpha_steps)))
                Gaussian_FWHM = st.number_input("Gaussian FWHM", value=0.1, min_value=0.005, step=0.005, format="%.3f")
            with col9:
                st.session_state.params["PO_toggle"] = st.checkbox("Preferred Orientation", value=False)
                # Always define po_model so the simulation calls (which now pass it explicitly)
                # never hit a NameError when PO is off; it is only used when PO is toggled on.
                po_model = None
                if st.session_state.params.get("PO_toggle"):
                    po_model = st.selectbox("PO Model:",["March-Dollase"])
                    #po_model = st.text_input("PO Model", value="March-Dollase")
                    if po_model == "March-Dollase":
                        POD_hkl_input = st.text_input("POD hkl", value="110")
                        #Convert hkl_POD to tuple
                        if len(POD_hkl_input) != 3 or not POD_hkl_input.isdigit():
                            st.write("hkl of POD must be three digets.")
                            st.session_state.params["hkl_POD"] = (0,0,1)
                        else:
                            st.session_state.params["hkl_POD"] = tuple(map(int, POD_hkl_input))
                        st.session_state.params["baseline"] = st.number_input("Baseline (between 0 and 1)", value=0.0, step=0.1, format="%.2f")
                        st.session_state.params["R"] = st.number_input("R", value=0.2, step=0.1, format="%.3f")
                        st.session_state.params["tau"] = st.number_input("tau (deg)", value=0.0, step=5.0, format="%.1f")
                        st.session_state.params["omega"] = st.number_input("omega (deg)", value=0.0, step=5.0, format="%.1f")
                        st.session_state.params["weight"] = st.number_input("weight", value=1.0, step=0.1, format="%.1f")
                    else:
                        st.write("{} model is not supported".format(po_model))
                        st.write("Choose from below:")
                        st.write("March-Dollase")
            lattice_params = {
                "a_val" : st.session_state.params.get("a_val"),
                "b_val" : st.session_state.params.get("b_val"),
                "c_val" : st.session_state.params.get("c_val"),
                "alpha" : st.session_state.params.get("alpha"),
                "beta" : st.session_state.params.get("beta"),
                "gamma" : st.session_state.params.get("gamma"),
            }
            wavelength = st.session_state.params.get("wavelength")
            chi = st.session_state.params.get("chi")
            # Dynamically build the list of sigma_ij keys present in params
            sigma_keys = ['sigma_11','sigma_22','sigma_33','sigma_12','sigma_13','sigma_23']
            sigma_params = {}
            for key in sigma_keys:
                sigma_params[key] = st.session_state.params.get(key)

            # Persist the full simulation context so the Refinement tab can gate on
            # it and reuse the exact model set up here (pure UI-state; no compute change).
            st.session_state.sim_context = {
                "selected_hkls": selected_hkls,
                "intensities": intensities,
                "symmetry": symmetry,
                "wavelength": wavelength,
                "chi": chi,
                "cijs": cijs,
                "lattice_params": lattice_params,
                "sigma_params": sigma_params,
                "po_model": po_model,
                # PO settings (Stage 2 refines these; hkl_POD stays fixed here).
                "PO_toggle": st.session_state.params.get("PO_toggle", False),
                "hkl_POD": st.session_state.params.get("hkl_POD", (0, 0, 1)),
                "po_values": {
                    "R": st.session_state.params.get("R", 1.0),
                    "tau": st.session_state.params.get("tau", 0.0),
                    "omega": st.session_state.params.get("omega", 0.0),
                    "weight": st.session_state.params.get("weight", 1.0),
                    "baseline": st.session_state.params.get("baseline", 0.0),
                },
            }

            # psi_steps / phi_steps / alpha_steps come directly from the sidebar widgets
            results_dict = {}  # Store results per HKL reflection
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Execute Calculations")
                #---------------------         
                #Generating epsilon-psi curves
                #--------------------- 
                epsilon_psi_dict = None
                if st.button("ε-ψ Curves") and selected_hkls:
                    epsilon_psi_dict = generate_epsilon_psi_curves(selected_hkls, psi_steps, phi_steps, alpha_steps, intensities=intensities, symmetry=symmetry, lattice_params=lattice_params, wavelength=wavelength, cijs=cijs, sigma_params=sigma_params, chi=chi, po_model=po_model)

                #Format the data and save to session_state
                if epsilon_psi_dict is not None:
                    if st.session_state.download_format == "Excel (.xlsx)":
                        output_buffer = io.BytesIO()
        
                        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                            for hkl_label, df in epsilon_psi_dict.items():
                                sheet_name = f"hkl_{hkl_label}"
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
                                worksheet = writer.sheets[sheet_name]
                                for i, col in enumerate(df.columns):
                                    # Measure data width on non-NaN values only; fall back to 0 if all NaN
                                    max_width = max(
                                        df[col].dropna().astype(str).map(len).max(),
                                        len(col)
                                    ) + 2
                                    worksheet.set_column(i, i, max_width)
        
                        output_buffer.seek(0)
                        datasource = epsilon_psi_dict
                        key = "epsilon_psi"
                        buffer = output_buffer
                        filename = "epsilon_psi.xlsx"
                        mime =("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        store_download(key, datasource, buffer, filename, mime)
        
                    elif st.session_state.download_format == "OpenDocument (.ods)":
                        output_buffer = io.BytesIO()
        
                        with pd.ExcelWriter(output_buffer, engine='odf') as writer:
                            for hkl_label, df in epsilon_psi_dict.items():
                                df.to_excel(writer, sheet_name=f"hkl_{hkl_label}", index=False)
        
                        output_buffer.seek(0)
                        datasource = epsilon_psi_dict
                        key = "epsilon_psi"
                        buffer = output_buffer
                        filename = "epsilon_psi.ods"
                        mime =("application/vnd.oasis.opendocument.spreadsheet")
                        store_download(key, datasource, buffer, filename, mime)

                    elif st.session_state.download_format == "ZIP of CSVs (.zip)":
                        output_buffer = io.BytesIO()

                        with zipfile.ZipFile(output_buffer, "w") as zf:
                            for hkl_label, df in epsilon_psi_dict.items():
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                zf.writestr(f"{hkl_label}.csv", csv_buffer.getvalue())
                    
                        output_buffer.seek(0)
                        datasource = epsilon_psi_dict
                        key = "epsilon_psi"
                        buffer = output_buffer
                        filename = "epsilon_psi.zip"
                        mime =("application/zip")
                        store_download(key, datasource, buffer, filename, mime)
                    else:
                        pass
    
                    st.success("File available for download above")
            
                #---------------------         
                #Generating cake plots
                #---------------------  
                if st.button("Cake Plot") and selected_hkls:
                    cake_dict = cake_data(selected_hkls, intensities, symmetry, lattice_params, 
                                                        wavelength, cijs, sigma_params, chi, po_model=po_model)
                    generate_cake_figures(cake_dict, selected_hkls, Funamori_broadening, chi=chi)
                
                    if cake_dict != {}:
                        #Format the data and save to session_state
                        if st.session_state.download_format == "Excel (.xlsx)":
                            output_buffer = io.BytesIO()
            
                            with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                                for hkl_label, df in cake_dict.items():
                                    sheet_name = f"hkl_{hkl_label}"
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
                                    worksheet = writer.sheets[sheet_name]
                                    for i, col in enumerate(df.columns):
                                        max_width = max(
                                            df[col].dropna().astype(str).map(len).max(),
                                            len(col)
                                        ) + 2
                                        worksheet.set_column(i, i, max_width)
            
                            output_buffer.seek(0)
                            datasource = cake_dict
                            key = "cake"
                            buffer = output_buffer
                            filename = "cake.xlsx"
                            mime =("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            store_download(key, datasource, buffer, filename, mime)
            
                        elif st.session_state.download_format == "OpenDocument (.ods)":
                            output_buffer = io.BytesIO()
            
                            with pd.ExcelWriter(output_buffer, engine='odf') as writer:
                                for hkl_label, df in cake_dict.items():
                                    df.to_excel(writer, sheet_name=f"hkl_{hkl_label}", index=False)
            
                            output_buffer.seek(0)
                            datasource = cake_dict
                            key = "cake"
                            buffer = output_buffer
                            filename = "cake.ods"
                            mime =("application/vnd.oasis.opendocument.spreadsheet")
                            store_download(key, datasource, buffer, filename, mime)
    
                        elif st.session_state.download_format == "ZIP of CSVs (.zip)":
                            output_buffer = io.BytesIO()
    
                            with zipfile.ZipFile(output_buffer, "w") as zf:
                                for hkl_label, df in cake_dict.items():
                                    csv_buffer = io.StringIO()
                                    df.to_csv(csv_buffer, index=False)
                                    zf.writestr(f"{hkl_label}.csv", csv_buffer.getvalue())
                        
                            output_buffer.seek(0)
                            datasource = cake_dict
                            key = "cake"
                            buffer = output_buffer
                            filename = "cake.zip"
                            mime =("application/zip")
                            store_download(key, datasource, buffer, filename, mime)
                        else:
                            pass
        
                        st.success("File available for download above")

                #Plotting preferred orientation
                if st.session_state.params.get("PO_toggle"):
                    st.subheader("Preferred Orientation")
                    if st.button("Plot PO Model"):
                        components = [
                            {"tau": st.session_state.params.get("tau"), "omega": st.session_state.params.get("omega"),"R": st.session_state.params.get("R") , "weight" : st.session_state.params.get("weight")
                            }
                        ]
                        hkl_POD = st.session_state.params.get("hkl_POD")
                        PO_MODEL = PO.PO_Model(po_model=po_model,
                                               components=components,
                                               baseline=st.session_state.params.get("baseline"),
                                               chi_deg = chi,
                                               POD_xtal=hkl_POD
                                              )
                        fig = PO_MODEL.make_intensity_pole_figure()
                        st.pyplot(fig)

                    if st.button("TEST PO Model"):
                        components = [
                        {"tau": st.session_state.params.get("tau"), "omega": st.session_state.params.get("omega"),"R": st.session_state.params.get("R") , "weight" : st.session_state.params.get("weight")
                        }
                        ]
                        hkl_POD = st.session_state.params.get("hkl_POD")
                        PO_MODEL = PO.PO_Model(po_model=po_model,
                                               components=components,
                                               baseline=st.session_state.params.get("baseline"),
                                               chi_deg = chi,
                                               POD_xtal=hkl_POD
                                              )
                        phi = np.linspace(0,360,32)
                        delta = np.linspace(-180,180,32)
                        I_grid, delta_grid, phi_grid = PO_MODEL.intensity_for_hkl((1,0,0), phi, delta)
                        #Plot the intensity distribution
                        fig = plt.figure(figsize=(6, 4))
                        ax_3d = fig.add_subplot(111, projection='3d')
                        ax_3d.view_init(elev=30, azim=-30)
                        surf = ax_3d.plot_surface(
                        delta_grid, phi_grid, I_grid,
                        cmap='viridis', edgecolor='k', alpha=0.9
                        )

                        ax_3d.set_xlabel("delta")
                        ax_3d.set_ylabel("phi")
                        ax_3d.set_zlabel("intensity")
                        st.pyplot(fig)
            
                st.subheader("Generate XRD patterns")
                if st.button("1D-XRD") and selected_hkls:
                    phi_values = np.radians(np.arange(0, 360, 2))
                    psi_values = 0
                    strain_sim_params = (symmetry, lattice_params, wavelength, cijs, sigma_params, chi, phi_values, psi_values)

                    XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, broadening=Funamori_broadening, po_model=po_model)
                    #Same global scaling the refinement applies, so the plotted pattern
                    #matches the value in the multiplier box
                    XRD_df = XRD_df.assign(**{"Total Intensity":
                        XRD_df["Total Intensity"]
                        * float(st.session_state.get("intensity_global_multiplier", 1.0))})

                    generate_1D_XRD_plot(XRD_df)

                    twotheta_grid = XRD_df["2th"]
                    total_pattern = XRD_df["Total Intensity"]

                    #Prepare .xy file
                    # .xy format is two columns, 2th and intensity
                    output_buffer = io.StringIO()
                    for tth, intensity in zip(twotheta_grid, total_pattern):
                        output_buffer.write(f"{tth:.5f} {intensity:.5f}\n")
                
                    # Move cursor to start for reading
                    output_buffer.seek(0)
                    datasource = XRD_df
                    key = "1D XRD"
                    buffer = output_buffer.getvalue()
                    filename = "1D XRD.xy"
                    mime =("text/plain")
                    store_download(key, datasource, buffer, filename, mime)
                    st.success("File available for download above")

                if poni_file is not None:
                    if st.button("2D-XRD") and selected_hkls:
                        # Save to a temporary file
                        with tempfile.NamedTemporaryFile(suffix=".poni") as tmp:
                            tmp.write(poni_file.read())
                            tmp.flush()
                        
                            # Load the geometry
                            ai = AzimuthalIntegrator()
                            ai.load(tmp.name)
                    
                        #Compute the cake data
                        cake_dict = {}
                        cake_dict = cake_data(selected_hkls, intensities, symmetry, lattice_params, 
                                                wavelength, cijs, sigma_params, chi, po_model=po_model)
                        cake_two_thetas, cake_deltas, cake_intensity = cake_dict_to_2Dcake(cake_dict, broadening=Funamori_broadening)

                        fig, ax = plt.subplots()
                    
                        im = ax.imshow(
                            cake_intensity.T,
                            extent=[cake_two_thetas.min(), cake_two_thetas.max(),
                                    cake_deltas.min(), cake_deltas.max()],
                            aspect='auto', 
                            origin='lower',
                            vmin=0,
                            vmax=np.percentile(cake_intensity, 98),
                            cmap='binary_r'
                        )

                        ax.set_xlabel("2θ (degrees)")
                        ax.set_ylabel("δ (degrees)")
                        ax.set_title("Cake")
                        plt.colorbar(im, ax=ax, label="Intensity")
                        st.pyplot(fig)

                        # Generate the raw detector image
                        # convert two_th to radians (requirement of pyFAI)
                        delta_axis_rad = np.deg2rad(cake_deltas)
                        tth_axis_rad = np.deg2rad(cake_two_thetas)

                        poni_file.seek(0)
                        text = poni_file.read().decode("utf-8")
                        # Parse line by line
                        for line in text.splitlines():
                            if "Detector_config" in line:
                                # Find the part after "max_shape"
                                idx = line.find("max_shape")
                                if idx != -1:
                                    # Example: max_shape: [2048, 2048]
                                    start = line.find("[", idx)
                                    end = line.find("]", idx)
                                    if start != -1 and end != -1:
                                        shape_str = line[start+1:end]  # '2048, 2048'
                                        height, width = map(int, shape_str.split(","))
                                        st.write("Detector size:", height, width)

                        det_shape = (height, width)  # (height, width)
                    
                        # Step 1: Compute angular field for each detector pixel
                        tth_pix = ai.twoThetaArray((height, width))  # radians
                        chi_pix = ai.chiArray((height, width))       # radians
                    
                        # Step 2: Build interpolator from cake space
                        interp = RegularGridInterpolator(
                            (tth_axis_rad, delta_axis_rad),
                            cake_intensity,
                            bounds_error=False,
                            fill_value=0
                        )

                        # Step 3: Sample cake intensities at detector angular coordinates
                        coords = np.stack([tth_pix, chi_pix], axis=-1)
                        det_image = interp(coords)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(det_image, 
                                       origin='lower', 
                                       cmap='binary_r', 
                                       aspect='equal', 
                                       vmin=0, 
                                       vmax=np.percentile(det_image, 98))
                        fig.colorbar(im, ax=ax, label='Intensity')
                        ax.set_xlabel('Pixel X')
                        ax.set_ylabel('Pixel Y')
                        st.pyplot(fig)
                
                #Make batch processing section
                if batch_upload:
                    parameters_df, results_df, results_blocks = batch_XRD(batch_upload, selected_hkls=selected_hkls, intensities=intensities, Gaussian_FWHM=Gaussian_FWHM, Funamori_broadening=Funamori_broadening, po_model=po_model)

                    #Plot up the data
                    fig, ax = plt.subplots(figsize=(10, 6))

                    #Get the first y dataset to compute the offset
                    y_initial = results_df["Intensity_iter1"]
                    y_offset = 0
                    offset_step = np.max(y_initial)*0.5
                
                    for idx in range(len(results_blocks)):
                        x_col = f"2th_iter{idx+1}"
                        y_col = f"Intensity_iter{idx+1}"
                        x = results_df[x_col]
                        y = results_df[y_col]
                        ax.plot(x, y + y_offset, color="black", lw=1, label=f"Iteration {idx+1}")
                        #Increase the offset
                        y_offset = y_offset+offset_step
                    
                    ax.set_xlabel("2θ (degrees)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.set_title("Batch XRD")
                    plt.tight_layout()
                    #Display the plot
                    st.pyplot(fig)
                
                    # Now you have two parts: parameters_df and results_df
                    # Export format: parameters first, then results
                    st.subheader("Download Computed Data")
                    output_buffer = io.BytesIO()
                    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                        parameters_df.to_excel(writer, sheet_name="Parameters", index=False)
                        results_df.to_excel(writer, sheet_name="Results", index=False)

                        # Auto-width adjustment for Parameters sheet
                        worksheet_params = writer.sheets["Parameters"]
                        for i, col in enumerate(parameters_df.columns):
                            max_width = max(parameters_df[col].astype(str).map(len).max(), len(str(col))) + 2
                            worksheet_params.set_column(i, i, max_width)

                        # Auto-width adjustment for "Results" sheet
                        worksheet = writer.sheets["Results"]
                        for i, col in enumerate(results_df.columns):
                            max_width = max(results_df[col].astype(str).map(len).max(), len(str(col))) + 2
                            worksheet.set_column(i, i, max_width)

                    output_buffer.seek(0)
            
                    st.download_button(
                        label="📥 Download Batch XRD as Excel (.xlsx)",
                        data=output_buffer,
                        file_name="XRD_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
                    #st.write("Parameters", parameters_df)
                    #st.write("Results", results_df)

        ### XRD Comparison/Refinement ----------------------------------------------------------------
        with col2:
            st.subheader("Overlay/refine with XRD")
            uploaded_XRD = st.file_uploader("Upload .xy experimental XRD file", type=[".xy"])

        if uploaded_XRD is not None:
            raw_lines = uploaded_XRD.read().decode("utf-8", errors="replace").splitlines()
            data_lines = [line for line in raw_lines if not line.strip().startswith("#") and line.strip()]
            data = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r'\s+', header=None, names=['2th', 'intensity'])
            x_exp = data['2th'].values
            y_exp = data['intensity'].values
            #Normalise exp data
            y_exp = y_exp/ np.max(y_exp)*100

            with col2:
                if st.button("Overlay XRD"):
                    phi_values = np.radians(np.arange(0, 360, 2))
                    psi_values = 0
                    #t = st.session_state.params.get("sigma_33") - st.session_state.params.get("sigma_11")
                    strain_sim_params = (symmetry, lattice_params, wavelength, cijs, sigma_params, chi, phi_values, psi_values)
                    XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, Funamori_broadening, po_model=po_model)
                    #Scale by the multiplier before overlaying, so what is drawn against
                    #the data is the same pattern the refinement would compare
                    XRD_df = XRD_df.assign(**{"Total Intensity":
                        XRD_df["Total Intensity"]
                        * float(st.session_state.get("intensity_global_multiplier", 1.0))})
                    generate_1D_XRD_overlay(XRD_df, x_exp, y_exp)
        
                #Construct the default parameter dictionary for refinement. The
                #multiplier's value comes from its box beside the hkl intensities above.
                _igm_val = float(st.session_state.get("intensity_global_multiplier", 1.0))
                other = {"chi" : chi,
                         "intensity_global_multiplier": _igm_val}

                setup_refinement_toggles(lattice_params, symmetry=symmetry, cijs=cijs, stress=sigma_params, other=other)
                # Scaling every hkl intensity is the same as scaling the whole pattern, so
                # refining both at once leaves the fit under-determined.
                _flags = st.session_state.get("refine_flags", {})
                if _flags.get("intensity_global_multiplier") and _flags.get("peak_intensity"):
                    st.warning(
                        "**intensity_global_multiplier** and **peak intensities** are "
                        "refining together — they are exactly degenerate (scaling every "
                        "hkl intensity by k is the same as scaling the pattern by k), so "
                        "the fit is under-determined and the uncertainties will be "
                        "meaningless. Refine the global multiplier first to set the "
                        "overall level, then hold it and unlock the individual "
                        "intensities for relative adjustments.")
            
                if st.button("Refine XRD"):
                    phi_values = np.radians(np.arange(0, 360, 10))
                    psi_values = 1
                
                    result = run_refinement(st.session_state.ref_params, st.session_state.refine_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, 
                                            phi_values, psi_values, wavelength, symmetry, x_exp, y_exp, lattice_params, cijs,
                                            sigma_params, chi, Funamori_broadening, po_model=po_model)
            
                    if result.success:
                        st.success("Refinement successful!")
                        #Write the refined multiplier back into its input box
                        if "intensity_global_multiplier" in result.params:
                            st.session_state.intensity_global_multiplier = \
                                float(result.params["intensity_global_multiplier"].value)
                        # Extract refined values from result.params
                        for key in st.session_state.params:
                            if key in result.params:
                                st.session_state.params[key] = result.params[key].value
                            else:
                                #Update the other lattice parameters that dont get refined for cubic etc
                                if key in ["b_val", "c_val"]:
                                    if symmetry == "cubic":
                                        st.session_state.params[key] = result.params["a_val"].value
                                    elif symmetry in ["hexagonal", "tetragonal_A", "tetragonal_B", "trigonal_A"]:
                                        if key == "b_val":
                                            st.session_state.params[key] = result.params["a_val"].value
                    
                        #Update the t and sigma values
                        t_opt = result.params["t"]
                        st.session_state.params["sigma_11"] = -t_opt / 3
                        st.session_state.params["sigma_22"] = -t_opt / 3
                        st.session_state.params["sigma_33"] = 2 * t_opt / 3
                        st.session_state.params["sigma_12"] = result.params["sigma_12"].value
                        st.session_state.params["sigma_13"] = result.params["sigma_13"].value
                        st.session_state.params["sigma_23"] = result.params["sigma_23"].value
    
                        #Update the intensity widgets and state values
                    
                        for key in st.session_state.intensities:
                            if key in result.params:
                                st.session_state.intensities[key] = result.params[key].value
                    
                        intensities = []
                        for i in selected_indices: 
                            intensities.append(st.session_state.intensities[f"intensity_{i}"])

                        #Ensure the parameters are updated for the plot
                        lattice_params = {
                            "a_val" : st.session_state.params.get("a_val"),
                            "b_val" : st.session_state.params.get("b_val"),
                            "c_val" : st.session_state.params.get("c_val"),
                            "alpha" : st.session_state.params.get("alpha"),
                            "beta" : st.session_state.params.get("beta"),
                            "gamma" : st.session_state.params.get("gamma"),
                        }
                        wavelength = st.session_state.params.get("wavelength")
                        chi = st.session_state.params.get("chi")
                        for key in sigma_keys:
                            sigma_params[key] = st.session_state.params.get(key)
                        c_keys = [key for key in st.session_state.params.keys() if key.startswith('c') and key not in ["c_val", "chi"]]
                        cijs = {}
                        for key in c_keys:
                            cijs[key] = st.session_state.params.get(key)
                            
                        st.markdown("### Fit Report")
                        report_str = fit_report(result)
                        st.code(report_str)
        
                        # Pack parameters for Generate_XRD
                        strain_sim_params = (
                            symmetry,
                            lattice_params,
                            wavelength,
                            cijs,
                            sigma_params,
                            chi,
                            phi_values,
                            psi_values
                        )
                    
                        XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params, Funamori_broadening, po_model=po_model)
                        #Show the pattern the fit actually compared against
                        _mult = float(result.params["intensity_global_multiplier"].value) \
                            if "intensity_global_multiplier" in result.params else 1.0
                        XRD_df = XRD_df.assign(**{"Total Intensity":
                                                  XRD_df["Total Intensity"] * _mult})
                        if abs(_mult - 1.0) > 1e-9:
                            st.caption(f"Simulated pattern scaled by the refined "
                                       f"intensity_global_multiplier = {_mult:.4g}.")
                        #twoth_sim = XRD_df["2th"]
                        #intensity_sim = XRD_df["Total Intensity"]
                        #x_min_sim = np.min(twoth_sim)
                        #x_max_sim = np.max(twoth_sim)
                        #mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
                        #x_exp_common = x_exp[mask]
                        #y_exp_common = y_exp[mask]
                        #interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=0)
                        #y_sim_common = interp_sim(x_exp_common)
        
                        #plot_overlay(x_exp_common, y_exp_common, x_exp_common, y_sim_common, title="Refined Fit")
                        generate_1D_XRD_overlay(XRD_df, x_exp, y_exp)

                    else:
                        st.error("Refinement failed.")

with tab_refine:
    st.header("Stage 1 Refinement — peak positions")
    with st.expander("How Stage 1 refinement works", expanded=False):
        st.markdown(cp.REFINEMENT_HELP_MD)

    _sc = st.session_state.get("sim_context")
    # Engine defaults, so the later stages still work when Stage 1's controls (which live
    # inside its data-loaded branch, and may override these) have not been rendered.
    _method, _maxnfev, _bounds, _latfrac = "leastsq", 200, {}, cp.LATTICE_BOUND_FRAC
    # Shared by all three stages, so each renders (with its own gating) regardless of
    # how far the previous one got.
    _peaks_df = None
    _sim_labels = [cp.hkl_label(h) for h in _sc["selected_hkls"]] if _sc else []
    if not _sc or not _sc.get("selected_hkls"):
        st.info("Set up the simulation first: on the **Simulation** tab load the elastic/hkl "
                "CSV and set the symmetry, χ, wavelength and (optional) PO model. Those "
                "values seed this refinement.")
    else:
        # --- 1) Experimental peak positions: CSV upload or the in-session extracted peaks ---
        _src = st.radio("Experimental peak source",
                        ["Upload peak-fit CSV", "Use current session peaks"], horizontal=True)
        _peaks_df = None
        if _src == "Upload peak-fit CSV":
            _rf = st.file_uploader("Peak-fit CSV (azimuth, 2th, hkl, …)", type=["csv"],
                                   key="refine_csv")
            if _rf is not None:
                try:
                    _peaks_df = cp.load_labelled_peaks_csv(_rf, filename=_rf.name)
                except Exception as _e:
                    st.error(f"Could not read CSV: {_e}")
        else:
            _sess = st.session_state.get("extracted_peaks_final")
            if _sess is not None and not _sess.empty and "hkl" in _sess.columns:
                _tmp = _sess.copy()
                _tmp["hkl"] = _tmp["hkl"].map(cp.normalize_hkl_label)
                _tmp = _tmp[_tmp["hkl"] != ""]
                _peaks_df = _tmp if not _tmp.empty else None
                if _peaks_df is None:
                    st.warning("Session peaks have no assigned hkl labels — assign them on the "
                               "**Peak Extraction** tab first.")
            else:
                st.warning("No labelled in-session peaks. Extract & label them on the "
                           "**Peak Extraction** tab, or upload a CSV.")

        if _peaks_df is None or _peaks_df.empty:
            st.caption("Load experimental peak positions to begin.")
        else:
            _exp_all = cp.experimental_hkl_curves(_peaks_df)
            _sim_labels = [cp.hkl_label(h) for h in _sc["selected_hkls"]]
            _matched = [L for L in _exp_all if L in _sim_labels]
            _unmatched = [L for L in _exp_all if L not in _sim_labels]
            if not _matched:
                st.warning(f"No data hkl labels match the simulated set {_sim_labels}. "
                           "Check the hkl labels assigned on the Peak Extraction tab.")
            else:
                if _unmatched:
                    st.caption("Ignoring unmatched hkls (not in the Simulation list): "
                               + ", ".join(_unmatched))

                # --- 2) hkl selection: refinement set and plotted set are independent ---
                _sel_c = st.columns([1, 1, 2])
                with _sel_c[0]:
                    st.markdown("**Refine**")
                    st.caption("hkls used in the fit")
                    _included = []
                    for _L in _matched:
                        _npts = int(_exp_all[_L]["n"].sum())
                        if st.checkbox(f"{_L} ({_npts})", value=True, key=f"refine_inc_{_L}"):
                            _included.append(_L)
                with _sel_c[1]:
                    st.markdown("**Plot**")
                    st.caption("hkls shown below")
                    _shown = []
                    for _L in _matched:
                        if st.checkbox(_L, value=True, key=f"refine_show_{_L}"):
                            _shown.append(_L)
                _exp_fit = {_L: _exp_all[_L] for _L in _included}
                _exp_view = {_L: _exp_all[_L] for _L in _shown}
                # Evaluate over the union so per-hkl RMSE is reported for anything either
                # fitted or plotted (the fit itself still uses only the "Refine" set).
                _exp_eval = {_L: _exp_all[_L] for _L in _matched
                             if _L in _included or _L in _shown}

                # --- 3) Parameters: initial value + refine toggle ---
                st.markdown("**Refine parameters** — set the initial value and tick to refine. "
                            "A robust first pass is `a`, `σ₁₁`, `σ₃₃` with the other stresses 0.")
                _lat_names = cp.lattice_param_names(_sc["symmetry"])
                _init, _flags = {}, {}
                _pc = st.columns(3)
                with _pc[0]:
                    st.caption("Lattice (Å)")
                    for _nm in ["a_val", "b_val", "c_val"]:
                        if _nm in _lat_names:
                            _init[_nm] = st.number_input(
                                _nm, value=float(_sc["lattice_params"][_nm]),
                                step=0.001, format="%.4f", key=f"ref_{_nm}")
                            _flags[_nm] = st.checkbox(f"refine {_nm}", value=(_nm == "a_val"),
                                                      key=f"reff_{_nm}")
                        else:
                            _init[_nm] = float(_sc["lattice_params"][_nm])
                    # Show which lengths are symmetry-locked to a (b/c inherit a).
                    _dep = [_n.replace("_val", "") for _n in ("b_val", "c_val")
                            if _n not in _lat_names]
                    if _dep:
                        st.caption(f"{', '.join(_dep)} = a  ({_sc['symmetry']})")
                with _pc[1]:
                    st.caption("Stress (GPa)")
                    for _nm, _on in [("sigma_11", True), ("sigma_33", True), ("sigma_22", False),
                                     ("sigma_12", False), ("sigma_13", False), ("sigma_23", False)]:
                        # Default the "rest" to 0; keep σ11/σ33 seeded from the Simulation tab.
                        _base = float(_sc["sigma_params"][_nm]) if _nm in ("sigma_11", "sigma_33") else 0.0
                        _init[_nm] = st.number_input(_nm, value=_base, step=0.1, format="%.3f",
                                                     key=f"ref_{_nm}")
                        _flags[_nm] = st.checkbox(f"refine {_nm}", value=_on, key=f"reff_{_nm}")
                with _pc[2]:
                    st.caption("Geometry / options")
                    _init["chi"] = st.number_input("chi (deg)", value=float(_sc["chi"]),
                                                   step=0.1, format="%.3f", key="ref_chi")
                    _flags["chi"] = st.checkbox("refine chi", value=False, key="reff_chi")
                    _fast = st.checkbox(
                        "Fast (coarse 12° δ grid)", value=False,
                        help="Use the coarse azimuth grid while iterating — faster, but the "
                             "modulation amplitude (t) can be slightly biased. Leave off for "
                             "the final refine.")
                    st.metric("t = σ₃₃ − σ₁₁ (initial)",
                              f"{_init['sigma_11']*-1 + _init['sigma_33']:.3f} GPa")

                # --- 3b) Engine & limits ---
                with st.expander("Refinement engine & limits", expanded=False):
                    st.markdown(cp.ENGINE_HELP_MD)
                    _ec = st.columns(3)
                    with _ec[0]:
                        _method = st.selectbox(
                            "Method", cp.REFINEMENT_METHODS, index=0,
                            help="lmfit minimiser. leastsq = Levenberg–Marquardt (gives "
                                 "uncertainties + correlations); least_squares = TRF "
                                 "(robust near bounds); nelder = simplex (no uncertainties).")
                    with _ec[1]:
                        _maxnfev = st.number_input(
                            "Max evaluations", min_value=10, max_value=100000, value=200,
                            step=50, help="Cap on forward-model calls (lmfit max_nfev). "
                                          "Raise if the fit stops before converging.")
                    with _ec[2]:
                        _latfrac = st.number_input(
                            "Lattice limit (± fraction)", min_value=0.01, max_value=5.0,
                            value=float(cp.LATTICE_BOUND_FRAC), step=0.05, format="%.2f",
                            help="Lattice a/b/c are bounded to ±this fraction of their "
                                 "STARTING value. A far-off initial guess therefore caps how "
                                 "far the refinement can move — widen this if a refined value "
                                 "lands on its limit.")
                    _bounds = {}
                    _free_now = [_n for _n, _on in _flags.items() if _on]
                    if _free_now:
                        st.caption("Limits applied to the refined parameters "
                                   "(edit to override):")
                        _blim = []
                        for _n in _free_now:
                            _lo, _hi = cp.param_bounds(_n, _init[_n], _latfrac)
                            _blim.append({"parameter": _n, "initial": _init[_n],
                                          "min": _lo, "max": _hi})
                        _bedit = st.data_editor(
                            pd.DataFrame(_blim), hide_index=True, use_container_width=True,
                            key="refine_bounds_editor",
                            column_config={
                                "parameter": st.column_config.TextColumn(disabled=True),
                                "initial": st.column_config.NumberColumn(disabled=True,
                                                                         format="%.4f"),
                                "min": st.column_config.NumberColumn(format="%.4f"),
                                "max": st.column_config.NumberColumn(format="%.4f")})
                        for _, _br in _bedit.iterrows():
                            _bounds[str(_br["parameter"])] = (float(_br["min"]),
                                                              float(_br["max"]))

                # --- 4) Compute model / run refinement (only on button press) ---
                _sig = (tuple(sorted(_included)), tuple(sorted(_shown)),
                        tuple((k, round(float(_init[k]), 6)) for k in sorted(_init)),
                        tuple(sorted(k for k, v in _flags.items() if v)), bool(_fast),
                        _method, int(_maxnfev),
                        tuple(sorted((k, round(v[0], 6), round(v[1], 6))
                                     for k, v in _bounds.items())))
                _b1, _b2 = st.columns(2)
                with _b1:
                    _preview = st.button("Preview model (no fit)", use_container_width=True)
                with _b2:
                    _run = st.button("Run Stage 1 refinement", type="primary",
                                     use_container_width=True)

                if _preview and _exp_eval:
                    with st.spinner("Evaluating model…"):
                        _ev = cp.evaluate_curves_and_residuals(compute_strain, _sc,
                                                               _exp_eval, _init, coarse=False)
                    st.session_state.stage1_view = {"eval": _ev, "sig": _sig, "result": None,
                                                    "init": dict(_init), "flags": dict(_flags)}
                if _run and _exp_fit:
                    with st.spinner("Refining…"):
                        _res = cp.run_stage1_refinement(
                            compute_strain, _sc, _exp_fit, _init, _flags, coarse=_fast,
                            max_nfev=int(_maxnfev), bounds=_bounds, method=_method,
                            lattice_frac=float(_latfrac))
                        _ev = cp.evaluate_curves_and_residuals(compute_strain, _sc, _exp_eval,
                                                               _res["values"], coarse=False)
                    st.session_state.stage1_view = {"eval": _ev, "sig": _sig, "result": _res,
                                                    "init": dict(_init), "flags": dict(_flags)}
                elif _run and not _exp_fit:
                    st.warning("Tick at least one hkl to include in the refinement.")

                _view = st.session_state.get("stage1_view")
                # --- 5) Results table ---
                if _view and _view.get("result"):
                    _res = _view["result"]
                    _vi, _vr = _view["init"], _res["values"]
                    (st.success if _res["success"] else st.warning)(
                        f"{_res['message']}  ·  {_res['n_free']} param(s), "
                        f"{_res['n_points']} points  ·  RMSE {_res['rmse']*1000:.2f} m°")
                    _rows = []
                    for _nm in [*_lat_names, *cp.SIGMA_NAMES, "chi"]:
                        if _view["flags"].get(_nm):
                            _rows.append({"parameter": _nm, "initial": _vi[_nm],
                                          "refined": _vr[_nm],
                                          "± 1σ": _res["errors"].get(_nm, float("nan"))})
                    _rows.append({"parameter": "t = σ₃₃−σ₁₁",
                                  "initial": _vi["sigma_33"] - _vi["sigma_11"],
                                  "refined": _res["t"], "± 1σ": float("nan")})
                    st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True,
                                 column_config={
                                     "initial": st.column_config.NumberColumn(format="%.4f"),
                                     "refined": st.column_config.NumberColumn(format="%.4f"),
                                     "± 1σ": st.column_config.NumberColumn(format="%.4f")})
                    if _res.get("at_limit"):
                        st.warning("Parameter(s) stopped **at a limit**: "
                                   + ", ".join(_res["at_limit"])
                                   + ". The refinement could not travel further — widen the "
                                     "limits under *Refinement engine & limits* (or start "
                                     "closer) and re-run.")
                    _stat_c = st.columns(4)
                    for _sc_i, (_lab, _key, _fmt) in enumerate([
                            ("reduced χ²", "redchi", "{:.3e}"), ("χ²", "chisqr", "{:.3e}"),
                            ("AIC", "aic", "{:.1f}"), ("fn evals", "nfev", "{:.0f}")]):
                        _val = _res.get(_key)
                        if _val is not None and np.isfinite(_val):
                            _stat_c[_sc_i].metric(_lab, _fmt.format(_val))
                    if _res.get("report"):
                        with st.expander(f"lmfit fit report ({_res.get('method', '')})",
                                         expanded=False):
                            st.code(_res["report"], language="text")
                            st.download_button("Download fit report (.txt)",
                                               _res["report"].encode("utf-8"),
                                               file_name="stage1_fit_report.txt",
                                               mime="text/plain")

                # --- 6) Overlay grid (data points + simulated line), 4 columns wide ---
                if _view and _view.get("eval"):
                    if _view["sig"] != _sig:
                        st.caption("⚠️ Inputs changed since the last compute — press **Preview** "
                                   "or **Run** to update the simulated curves.")
                    _sim_curves = _view["eval"]["sim_curves"]
                    _perhkl = _view["eval"]["per_hkl"]
                    if not _exp_view:
                        st.caption("No hkls ticked under **Plot** — nothing to display.")
                    else:
                        st.plotly_chart(
                            cp.plot_refinement_grid(_exp_view, _sim_curves, ncols=4,
                                                    included=set(_included)),
                            width='stretch')
                    _ph = pd.DataFrame([{"hkl": _L, "in fit": _L in _included,
                                         "plotted": _L in _shown,
                                         "points": int(_exp_all[_L]["n"].sum()),
                                         "RMSE (°)": _perhkl.get(_L, float("nan"))}
                                        for _L in _matched])
                    st.dataframe(_ph, hide_index=True, use_container_width=True,
                                 column_config={"RMSE (°)": st.column_config.NumberColumn(format="%.4f")})
                else:
                    st.caption("Press **Preview model** to overlay the current simulation on the "
                               "data, or **Run Stage 1 refinement** to fit.")

    # =====================================================================
    # STAGE 2 — preferred orientation from the azimuthal INTENSITY variation
    # =====================================================================
    st.divider()
    st.header("Stage 2 Refinement — preferred orientation")
    with st.expander("How Stage 2 refinement works", expanded=False):
        st.markdown(cp.STAGE2_HELP_MD)

    if not _sc or not _sc.get("selected_hkls"):
        st.info("Set up the simulation first on the **Simulation** tab.")
    elif not _sc.get("PO_toggle"):
        st.info("Enable **Preferred Orientation** on the Simulation tab (and set the "
                "POD hkl / model) to refine PO parameters here.")
    elif _peaks_df is None or _peaks_df.empty:
        st.info("Load experimental peak positions in the **Stage 1** section above — "
                "Stage 2 fits the intensities of those same extracted peaks.")
    else:
        _int_measure = st.radio(
            "Experimental intensity measure",
            ["Integrated area", "Fitted amplitude"], horizontal=True,
            help="Integrated area (from amp, FWHM and gl) is the physical "
                 "reflection intensity and is unbiased when texture broadens "
                 "the arcs; amplitude is the raw fitted peak height.")
        _measure = "area" if _int_measure.startswith("Integrated") else "amplitude"
        _int_all = cp.experimental_intensity_curves(_peaks_df, measure=_measure)
        _int_matched = [_L for _L in _int_all if _L in _sim_labels]
        if not _int_matched:
            st.warning("No hkl labels in the data match the simulated set.")
        else:
            _s2c = st.columns([1, 1, 2])
            with _s2c[0]:
                st.markdown("**Refine**")
                st.caption("hkls used in the fit")
                _inc2 = [_L for _L in _int_matched
                         if st.checkbox(f"{_L} ({int(_int_all[_L]['n'].sum())})",
                                        value=True, key=f"s2_inc_{_L}")]
            with _s2c[1]:
                st.markdown("**Plot**")
                st.caption("hkls shown below")
                _shown2 = [_L for _L in _int_matched
                           if st.checkbox(_L, value=True, key=f"s2_show_{_L}")]
            _fit2 = {_L: _int_all[_L] for _L in _inc2}
            _view2 = {_L: _int_all[_L] for _L in _shown2}
            _eval2 = {_L: _int_all[_L] for _L in _int_matched
                      if _L in _inc2 or _L in _shown2}

            # PO parameters: initial value + refine toggle
            st.markdown("**PO parameters** — refine in stages: `R` first, then "
                        "`tau`/`omega`, then `baseline` (see the help above).")
            _pov = dict(_sc.get("po_values") or {})
            _init2, _flags2 = {}, {}
            _pc2 = st.columns(4)
            _po_specs = [("R", 0.9, 0.05, "%.3f", True,
                          "March–Dollase strength. R = 1 is no texture; < 1 "
                          "platy, > 1 needle-like. Typical 0.3–1.5."),
                         ("tau", 0.0, 1.0, "%.2f", False,
                          "Tilt of the preferred-orientation axis (degrees)."),
                         ("omega", 0.0, 1.0, "%.2f", False,
                          "Rotation of the preferred-orientation axis (degrees)."),
                         ("baseline", 0.05, 0.05, "%.3f", False,
                          "Isotropic fraction added to the PO intensity (0–1). "
                          "Do not start it at exactly 0 (its limit) when "
                          "refining — a parameter sitting on a bound cannot "
                          "move. Typical 0.0–0.3.")]
            for _i2, (_nm, _dflt, _stp, _fmt, _on, _hlp) in enumerate(_po_specs):
                with _pc2[_i2]:
                    _init2[_nm] = st.number_input(
                        _nm, value=float(_pov.get(_nm, _dflt)), step=_stp,
                        format=_fmt, key=f"s2_v_{_nm}", help=_hlp)
                    _flags2[_nm] = st.checkbox(f"refine {_nm}", value=_on,
                                               key=f"s2_f_{_nm}")
            _init2["weight"] = float(_pov.get("weight", 1.0))

            # phi-integration sampling: auto from R, or forced.
            _nphi_c = st.columns([1, 3])
            with _nphi_c[0]:
                _nphi_in = st.number_input(
                    "φ sampling (0 = auto)", min_value=0, max_value=4096,
                    value=0, step=36, key="s2_nphi",
                    help="Number of φ points the PO surface is integrated over. "
                         "0 picks it automatically from R — sharper texture "
                         "(small or large R) needs more points. Override to check "
                         "the result is stable against the sampling.")
            _nphi = int(_nphi_in) or None
            _nphi_auto = cp.adaptive_n_phi(_init2["R"])
            with _nphi_c[1]:
                st.caption(
                    f"Auto for R = {_init2['R']:.3g} → **{_nphi_auto}** φ points "
                    f"(sharpness max(R, 1/R) = {max(_init2['R'], 1/_init2['R'] if _init2['R'] else 1):.2f})."
                    + ("" if _nphi is None else f"  Overridden to **{_nphi}**."))

            if (_flags2.get("tau") or _flags2.get("omega")) and \
                    abs(_init2["R"] - 1.0) < 1e-6:
                st.warning("`R` starts at exactly 1.0 (isotropic), where **tau and "
                           "omega have zero gradient** and cannot be refined. Refine "
                           "`R` alone first, or start it away from 1.0.")
            if _flags2.get("baseline") and _init2["baseline"] <= 1e-9:
                st.warning("`baseline` starts at exactly 0, which is its lower "
                           "limit — a parameter sitting **on a bound cannot move**. "
                           "Start it slightly above 0 (e.g. 0.05) to refine it.")

            _b3, _b4 = st.columns(2)
            with _b3:
                _prev2 = st.button("Preview PO model", use_container_width=True,
                                   key="s2_preview")
            with _b4:
                _run2 = st.button("Run Stage 2 refinement", type="primary",
                                  use_container_width=True, key="s2_run")
            _sig2 = (tuple(sorted(_inc2)), tuple(sorted(_shown2)), _measure,
                     tuple((k, round(float(v), 6)) for k, v in sorted(_init2.items())),
                     tuple(sorted(k for k, v in _flags2.items() if v)),
                     _method, int(_maxnfev), _nphi)

            if _prev2 and _eval2:
                with st.spinner("Evaluating PO model…"):
                    _ev2 = cp.evaluate_po_curves(_sc, _eval2, _init2, n_phi=_nphi)
                st.session_state.stage2_view = {"eval": _ev2, "sig": _sig2,
                                                "result": None, "init": dict(_init2),
                                                "flags": dict(_flags2)}
            if _run2 and _fit2:
                with st.spinner("Refining PO parameters…"):
                    _res2 = cp.run_stage2_refinement(
                        _sc, _fit2, _init2, _flags2, method=_method,
                        max_nfev=int(_maxnfev), n_phi=_nphi)
                    # Score/plot at the sampling the refined R actually needs.
                    _ev2 = cp.evaluate_po_curves(
                        _sc, _eval2, _res2["values"],
                        n_phi=_nphi or max(_res2["n_phi"],
                                           _res2["n_phi_suggested"]))
                st.session_state.stage2_view = {"eval": _ev2, "sig": _sig2,
                                                "result": _res2, "init": dict(_init2),
                                                "flags": dict(_flags2)}
            elif _run2 and not _fit2:
                st.warning("Tick at least one hkl to include in the refinement.")

            _v2 = st.session_state.get("stage2_view")
            if _v2 and _v2.get("result"):
                _r2 = _v2["result"]
                (st.success if _r2["success"] else st.warning)(
                    f"{_r2['message']}  ·  {_r2['n_free']} param(s), "
                    f"{_r2['n_points']} points  ·  RMSE {_r2['rmse']:.4g}")
                _rows2 = [{"parameter": _nm, "initial": _v2["init"][_nm],
                           "refined": _r2["values"][_nm],
                           "± 1σ": _r2["errors"].get(_nm, float("nan"))}
                          for _nm, *_ in _po_specs if _v2["flags"].get(_nm)]
                if _rows2:
                    st.dataframe(pd.DataFrame(_rows2), hide_index=True,
                                 use_container_width=True,
                                 column_config={
                                     "initial": st.column_config.NumberColumn(format="%.4f"),
                                     "refined": st.column_config.NumberColumn(format="%.4f"),
                                     "± 1σ": st.column_config.NumberColumn(format="%.4f")})
                if _r2.get("at_limit"):
                    st.warning("Parameter(s) stopped **at a limit**: "
                               + ", ".join(_r2["at_limit"]))
                if _r2.get("n_phi_suggested", 0) > _r2.get("n_phi", 0):
                    st.warning(
                        f"The refined R = {_r2['values']['R']:.3g} is sharper than "
                        f"the start, and needs **{_r2['n_phi_suggested']} φ points** "
                        f"— this fit used {_r2['n_phi']} (held fixed to keep the "
                        "gradients clean). **Run again** from these values to "
                        "refine at the finer sampling.")
                _sc2 = st.columns(4)
                _sc2[0].metric("global scale", f"{_r2['scale']:.4g}",
                               help=f"φ sampling used: {_r2.get('n_phi')} points")
                for _j, (_lab, _key, _fmt2) in enumerate(
                        [("reduced χ²", "redchi", "{:.3e}"),
                         ("AIC", "aic", "{:.1f}"), ("fn evals", "nfev", "{:.0f}")]):
                    _val2 = _r2.get(_key)
                    if _val2 is not None and np.isfinite(_val2):
                        _sc2[_j + 1].metric(_lab, _fmt2.format(_val2))
                if _r2.get("report"):
                    with st.expander(f"lmfit fit report ({_r2.get('method','')})",
                                     expanded=False):
                        st.code(_r2["report"], language="text")
                        st.download_button(
                            "Download fit report (.txt)",
                            _r2["report"].encode("utf-8"),
                            file_name="stage2_fit_report.txt", mime="text/plain",
                            key="s2_dl")
                st.caption("Apply these to the model by entering the refined values "
                           "in the PO controls on the **Simulation** tab.")

            if _v2 and _v2.get("eval"):
                if _v2["sig"] != _sig2:
                    st.caption("⚠️ Inputs changed since the last compute — press "
                               "**Preview PO model** or **Run** to update.")
                if not _view2:
                    st.caption("No hkls ticked under **Plot** — nothing to display.")
                else:
                    st.plotly_chart(
                        cp.plot_intensity_grid(_view2, _v2["eval"]["sim_curves"],
                                               ncols=4, included=set(_inc2)),
                        width='stretch')
                _ph2 = pd.DataFrame(
                    [{"hkl": _L, "in fit": _L in _inc2, "plotted": _L in _shown2,
                      "points": int(_int_all[_L]["n"].sum()),
                      "RMSE": _v2["eval"]["per_hkl"].get(_L, float("nan"))}
                     for _L in _int_matched])
                st.dataframe(_ph2, hide_index=True, use_container_width=True,
                             column_config={"RMSE": st.column_config.NumberColumn(format="%.4g")})
            else:
                st.caption("Press **Preview PO model** to overlay the current PO "
                           "model on the measured intensities, or **Run Stage 2 "
                           "refinement** to fit.")

    # =====================================================================
    # STAGE 3 — fit the background-subtracted IMAGE directly
    # =====================================================================
    st.divider()
    st.header("Stage 3 Refinement — image intensity")
    with st.expander("How Stage 3 refinement works", expanded=False):
        st.markdown(cp.STAGE3_HELP_MD)

    _img = st.session_state.get("subtracted_image")
    if not _sc or not _sc.get("selected_hkls"):
        st.info("Set up the simulation first on the **Simulation** tab.")
    elif _img is None:
        st.info("No subtracted image available. Compute or load a background on the "
                "**Cake Import & Background** tab, or upload a pre-subtracted cake on the "
                "**Peak Extraction** tab.")
    else:
        _s3x, _s3grid = _img
        _s3c = st.columns(4)
        with _s3c[0]:
            _s3_az = st.number_input(
                "Azimuth step (°)", min_value=float(cp.MIN_AZ_STEP), max_value=90.0,
                value=5.0, step=0.5, format="%.2f", key="s3_az",
                help="Azimuth box size for the comparison grid. Both the image and the "
                     "model are averaged into these boxes. Limited to ≥ {:g}° because the "
                     "simulation samples azimuth on a {:g}° grid — finer boxes could not "
                     "all be filled by the model.".format(cp.MIN_AZ_STEP, cp.MIN_AZ_STEP))
            _s3_tth = st.number_input(
                "2θ step (°)", min_value=0.002, max_value=1.0, value=0.02, step=0.005,
                format="%.3f", key="s3_tth",
                help="2θ box size for the comparison grid. Aim for 4–8 boxes across a "
                     "ring's FWHM — too coarse blurs the peak, too fine keeps the noise.")
            _s3_sub = st.number_input(
                "Sim sub-samples / 2θ box", min_value=1, max_value=50, value=5, step=1,
                key="s3_sub",
                help="The model is evaluated and convolved on a grid this many times "
                     "finer than the comparison boxes, then averaged down — so a coarse "
                     "comparison grid does not degrade the simulation's accuracy.")
        with _s3c[1]:
            _s3_k = st.number_input(
                "Window × width", min_value=1.0, max_value=20.0, value=4.0, step=0.5,
                key="s3_k",
                help="Window half-width as a multiple of the estimated ring width. The "
                     "window is fixed from the starting model so the residual keeps a "
                     "constant length.")
            _s3_nphi_in = st.number_input(
                "φ sampling (0 = auto)", min_value=0, max_value=4096, value=0, step=72,
                key="s3_nphi",
                help="Orientation samples over φ. Auto scales with R (sharper texture "
                     "needs more) with a floor of {}. φ is both the PO integration axis "
                     "and the source of the strain broadening here, and cost scales "
                     "linearly with it — so this is also the main speed control."
                     .format(cp.STAGE3_MIN_N_PHI))
        with _s3c[2]:
            st.markdown("**Include hkls**")
            st.caption("rings modelled and fitted")
            _s3_all = [cp.hkl_label(h) for h in _sc["selected_hkls"]]
            _s3_inc = [L for L in _s3_all
                       if st.checkbox(L, value=True, key=f"s3_inc_{L}")]
        with _s3c[3]:
            st.caption("Seeding")
            st.caption("Values below start from Stages 1–2 (or the Simulation tab). "
                       "Press to refresh them from the latest results.")
            _s3_reseed = st.button("↻ Reseed values", use_container_width=True,
                                   key="s3_reseed")

        # ---- Starting values: Stage 3 result if any, else Stages 1-2, else Simulation ----
        _s1v = (st.session_state.get("stage1_view") or {}).get("result")
        _s2v = (st.session_state.get("stage2_view") or {}).get("result")
        _s3v = (st.session_state.get("stage3_view") or {}).get("result")
        _s3_seed = {}
        for _k in ("a_val", "b_val", "c_val", "alpha", "beta", "gamma"):
            _s3_seed[_k] = float(_sc["lattice_params"][_k])
        for _k in cp.SIGMA_NAMES:
            _s3_seed[_k] = float(_sc["sigma_params"].get(_k, 0.0))
        _s3_seed["chi"] = float(_sc["chi"])
        _pov3 = dict(_sc.get("po_values") or {})
        for _k, _d in (("R", 0.9), ("tau", 0.0), ("omega", 0.0), ("baseline", 0.05),
                       ("weight", 1.0)):
            _s3_seed[_k] = float(_pov3.get(_k, _d))
        _s3_seed["fwhm"] = 0.10
        if _s1v:
            _s3_seed.update({k: v for k, v in _s1v["values"].items() if k in _s3_seed})
        if _s2v:
            _s3_seed.update({k: v for k, v in _s2v["values"].items()
                             if k in ("R", "tau", "omega", "baseline", "weight")})
        if _s3v:
            _s3_seed.update({k: v for k, v in _s3v["values"].items() if k in _s3_seed})
        _s3_src = ", ".join([s for s, ok in (("Stage 1", _s1v), ("Stage 2", _s2v),
                                             ("last Stage 3 fit", _s3v)) if ok]) \
            or "the Simulation tab"
        # Writing widget state before the widgets are built is safe; doing it after is not.
        if _s3_reseed:
            for _k, _v in _s3_seed.items():
                st.session_state[f"s3_v_{_k}"] = float(_v)
            st.rerun()

        st.markdown("**Parameters** — set the starting value and tick to refine. "
                    f"Seeded from {_s3_src}; refine a few at a time.")
        _s3_lat = cp.lattice_param_names(_sc["symmetry"])
        _s3_spec = [
            ("Lattice (Å)", [(p, 0.001, "%.5f") for p in _s3_lat]),
            ("Stress (GPa)", [(p, 0.1, "%.3f") for p in ("sigma_11", "sigma_33")]),
            ("Preferred orientation", [("R", 0.05, "%.3f"), ("tau", 1.0, "%.2f"),
                                       ("omega", 1.0, "%.2f"), ("baseline", 0.05, "%.3f")]),
            ("Profile / geometry", [("fwhm", 0.005, "%.4f"), ("chi", 0.1, "%.3f")]),
        ]
        _s3_default_on = {"fwhm"}
        _s3_init, _s3_flags = dict(_s3_seed), {}
        _pcols = st.columns(len(_s3_spec))
        for _ci, (_title, _params) in enumerate(_s3_spec):
            with _pcols[_ci]:
                st.caption(_title)
                for _p, _stp, _fmt in _params:
                    _s3_init[_p] = float(st.number_input(
                        _p, value=float(_s3_seed[_p]), step=_stp, format=_fmt,
                        key=f"s3_v_{_p}"))
                    _s3_flags[_p] = st.checkbox(f"refine {_p}", value=(_p in _s3_default_on),
                                                key=f"s3_f_{_p}")
        _s3_fwhm = _s3_init["fwhm"]
        # Only the ticked rings are modelled, so the windows and fit cover just those.
        _sc3 = dict(_sc)
        _sc3["selected_hkls"] = [h for h in _sc["selected_hkls"]
                                 if cp.hkl_label(h) in _s3_inc]
        _sc3["intensities"] = [i for h, i in zip(_sc["selected_hkls"], _sc["intensities"])
                               if cp.hkl_label(h) in _s3_inc]
        if _s3_flags.get("baseline") and _s3_init["baseline"] <= 1e-9:
            st.warning("`baseline` starts at its 0 limit and cannot move — start it "
                       "slightly above 0.")
        if _s3_flags.get("R") and abs(_s3_init["R"] - 1.0) < 1e-6 and (
                _s3_flags.get("tau") or _s3_flags.get("omega")):
            st.warning("`R` starts at exactly 1.0 (isotropic), where **tau and omega have "
                       "zero gradient**. Refine `R` alone first, or start it away from 1.0.")

        _b5, _b6 = st.columns(2)
        with _b5:
            _s3_prev = st.button("Preview image model", use_container_width=True,
                                 key="s3_preview")
        with _b6:
            _s3_run = st.button("Run Stage 3 refinement", type="primary",
                                use_container_width=True, key="s3_run")

        # --- Masking: drop boxes with no usable data from the residual ---
        # Gated by a checkbox rather than an expander: Streamlit fixes a plotly chart's
        # width at first render, and inside a collapsed expander that width is tiny, so
        # the mask editor would come back 60px wide and never resize.
        # Settings persist in session state, so the mask keeps applying while the editor
        # is hidden; the widgets below just overwrite these when it is open.
        _s3_excl = list(st.session_state.get("s3_exclude_rows", []))
        _s3_below = (float(st.session_state["s3_below_val"])
                     if st.session_state.get("s3_below_on") else None)
        _s3_above = (float(st.session_state["s3_above_val"])
                     if st.session_state.get("s3_above_on") else None)
        _s3_show_mask = st.checkbox(
            "🚫 Edit data mask (detector gaps / cut-off azimuths)", value=False,
            key="s3_show_mask")
        if not _s3_show_mask and _s3_excl:
            st.caption(f"Mask active: {len(_s3_excl)} exclusion region(s).")
        if _s3_show_mask:
            st.markdown(
                "Detector gaps and cut-off azimuth wedges must be kept out of the "
                "residual, or the fit chases empty boxes. **Each row has its own 2θ "
                "range, so the excluded azimuths can differ at different radii** — which "
                "is how detector shadows behave. Leave a 2θ bound blank to cover the "
                "whole range, and set `az_from` > `az_to` for a range that wraps through "
                "±180. Masking only takes effect inside the ring windows (shaded below), "
                "so a region drawn wider than those is truncated to them.")
            st.markdown("**Intensity thresholds** — mask boxes by value as well as by "
                        "region. Useful for dead areas that read near zero, or for hot "
                        "pixels and saturated spots.")
            _mk = st.columns(4)
            with _mk[0]:
                _below_on = st.checkbox("Mask below", value=False, key="s3_below_on",
                                        help="Exclude boxes whose mean value is BELOW "
                                             "the threshold (e.g. dead detector area).")
            with _mk[1]:
                _below_v = st.number_input("below threshold", value=0.0, step=1.0,
                                           format="%.4g", key="s3_below_val",
                                           label_visibility="collapsed")
            with _mk[2]:
                _above_on = st.checkbox("Mask above", value=False, key="s3_above_on",
                                        help="Exclude boxes whose mean value is ABOVE "
                                             "the threshold (e.g. saturated spots).")
            with _mk[3]:
                _above_v = st.number_input("above threshold", value=0.0, step=1.0,
                                           format="%.4g", key="s3_above_val",
                                           label_visibility="collapsed")
            _s3_below = float(_below_v) if _below_on else None
            _s3_above = float(_above_v) if _above_on else None
            # Draw a region straight onto the image. Streamlit reports the drag as data
            # coordinates; a heatmap yields no points, but the box bounds are all we need.
            st.caption("**Drag a box on the image** to add an exclusion region, or edit "
                       "the table directly. Existing regions are shaded cyan.")
            _mask_sel = st.plotly_chart(
                cp.plot_mask_editor(_s3x, _s3grid,
                                    st.session_state.get("s3_exclude_rows", []),
                                    percentile=cake_percentile,
                                    windows=st.session_state.get("s3_windows_deg")),
                key="s3_mask_editor", on_select="rerun", selection_mode="box",
                use_container_width=True)
            _new_reg = cp.selection_to_region((_mask_sel or {}).get("selection", {}))
            # Streamlit replays the same selection on every rerun, so only act on a change.
            if _new_reg and _new_reg != st.session_state.get("s3_last_box"):
                st.session_state.s3_last_box = _new_reg
                st.session_state.s3_exclude_rows = (
                    st.session_state.get("s3_exclude_rows", []) + [_new_reg])
                st.session_state.pop("s3_exclude_df", None)   # rebuild the table below
                st.rerun()
            _s3_regions = st.data_editor(
                st.session_state.get(
                    "s3_exclude_df",
                    pd.DataFrame(st.session_state.get("s3_exclude_rows", []),
                                 columns=cp.EXCLUDE_REGION_COLUMNS)),
                num_rows="dynamic", use_container_width=True, key="s3_exclude_editor",
                column_config={
                    "az_from": st.column_config.NumberColumn("azimuth from (°)", format="%.1f"),
                    "az_to": st.column_config.NumberColumn("azimuth to (°)", format="%.1f"),
                    "tth_from": st.column_config.NumberColumn("2θ from (°)", format="%.3f"),
                    "tth_to": st.column_config.NumberColumn("2θ to (°)", format="%.3f")})
            st.session_state.s3_exclude_df = _s3_regions
            _s3_excl = [r for r in _s3_regions.to_dict("records")
                        if pd.notna(r.get("az_from")) and pd.notna(r.get("az_to"))]
            # Table edits (including row deletion) are the source of truth for the shading.
            st.session_state.s3_exclude_rows = _s3_excl
            _mc = st.columns([3, 1])
            _mc[0].caption(f"{len(_s3_excl)} exclusion region(s) active."
                           if _s3_excl else "No exclusion regions — drag one above to add.")
            if _mc[1].button("Clear all", use_container_width=True, key="s3_clear_regions"):
                for _k in ("s3_exclude_rows", "s3_exclude_df", "s3_last_box"):
                    st.session_state.pop(_k, None)
                st.rerun()

        # compute_strain reads the PO parameters from st.session_state.params, not from its
        # arguments, so they must be written there before every forward-model call or a
        # refinement of R/tau/omega/baseline would leave the simulation unchanged.
        def _s3_po_apply(_v):
            _p = st.session_state.get("params")
            if _p is not None:
                for _k in ("R", "tau", "omega", "weight", "baseline"):
                    if _k in _v:
                        _p[_k] = float(_v[_k])

        if (_s3_prev or _s3_run) and not _s3_inc:
            st.warning("Tick at least one hkl to include.")
        elif _s3_prev or _s3_run:
            _s3_nphi = int(_s3_nphi_in) or None
            with st.spinner("Building the comparison grid…"):
                _seed_dfs = cp._stage3_sim_dfs(compute_strain, _sc3, _s3_init,
                                               n_phi=_s3_nphi, po_apply=_s3_po_apply)
                _s3_blocks = cp.build_stage3_grid(
                    _s3x, _s3grid, _seed_dfs, az_step=float(_s3_az),
                    tth_step=float(_s3_tth), fwhm=float(_s3_fwhm), roi_k=float(_s3_k),
                    n_sub=int(_s3_sub), exclude_regions=_s3_excl,
                    mask_below=_s3_below, mask_above=_s3_above)
            # Remember the windows so the mask editor can shade where masking applies.
            st.session_state.s3_windows_deg = {
                _L: (float(_s3_blocks.tth_edges[_c0]),
                     float(_s3_blocks.tth_edges[min(_c1, _s3_blocks.tth_edges.size - 1)]))
                for _L, (_c0, _c1) in _s3_blocks.windows.items()}
            if not _s3_blocks.labels:
                st.warning("No ring windows fell inside the image 2θ range.")
            else:
                if _s3_run:
                    with st.spinner("Refining against the image…"):
                        _s3_res = cp.run_stage3_refinement(
                            compute_strain, _sc3, _s3_blocks, _s3_init, _s3_flags,
                            method=_method, max_nfev=int(_maxnfev), n_phi=_s3_nphi,
                            po_apply=_s3_po_apply)
                        # Score/display at the sampling the refined R actually needs.
                        _s3_ev = cp.evaluate_stage3(
                            compute_strain, _sc3, _s3_blocks, _s3_res["values"],
                            _s3_res["values"].get("fwhm", _s3_init["fwhm"]),
                            n_phi=_s3_nphi or max(_s3_res["n_phi"],
                                                  _s3_res["n_phi_suggested"]),
                            po_apply=_s3_po_apply)
                else:
                    with st.spinner("Evaluating…"):
                        _s3_res = None
                        _s3_ev = cp.evaluate_stage3(compute_strain, _sc3, _s3_blocks,
                                                    _s3_init, float(_s3_fwhm),
                                                    n_phi=_s3_nphi,
                                                    po_apply=_s3_po_apply)
                st.session_state.stage3_view = {"blocks": _s3_blocks, "eval": _s3_ev,
                                                "result": _s3_res, "init": dict(_s3_init),
                                                "flags": dict(_s3_flags)}

        _v3 = st.session_state.get("stage3_view")
        if _v3:
            _bl, _ev3, _r3 = _v3["blocks"], _v3["eval"], _v3.get("result")
            _m3 = st.columns(4)
            _m3[0].metric("boxes compared", f"{_bl.n_points:,}",
                          help=f"grid {_bl.data.shape[0]} azimuth × {_bl.data.shape[1]} 2θ")
            _m3[1].metric("global scale", f"{_ev3['scale']:.4g}")
            _m3[2].metric("RMSE", f"{_ev3['rmse']:.4g}")
            _m3[3].metric("φ samples", f"{_ev3.get('n_phi', '—')}",
                          help=f"{len(_bl.labels)} ring(s); φ chosen from R unless overridden")
            _msk = int(_bl.masked_mask.sum())
            if _msk:
                _win = int(_bl.window_mask.sum()) if _bl.window_mask is not None else 0
                st.caption(f"🚫 {_msk:,} of {_win:,} in-window boxes masked "
                           f"({_msk / max(_win, 1) * 100:.1f}%) — shaded red below and "
                           "excluded from the scale and residual.")
            if _r3:
                (st.success if _r3["success"] else st.warning)(
                    f"{_r3['message']}  ·  {_r3['n_free']} param(s), "
                    f"{_r3['n_points']:,} blocks  ·  RMSE {_r3['rmse']:.4g}")
                _rows3 = [{"parameter": _n, "initial": _v3["init"][_n],
                           "refined": _r3["values"][_n],
                           "± 1σ": _r3["errors"].get(_n, float("nan"))}
                          for _n in _v3["flags"] if _v3["flags"][_n]]
                if _rows3:
                    st.dataframe(pd.DataFrame(_rows3), hide_index=True,
                                 use_container_width=True,
                                 column_config={
                                     "initial": st.column_config.NumberColumn(format="%.4f"),
                                     "refined": st.column_config.NumberColumn(format="%.4f"),
                                     "± 1σ": st.column_config.NumberColumn(format="%.4f")})
                if _r3.get("at_limit"):
                    st.warning("Parameter(s) stopped **at a limit**: "
                               + ", ".join(_r3["at_limit"]))
                if _r3.get("n_phi_suggested", 0) > _r3.get("n_phi", 0):
                    st.warning(
                        f"The refined R = {_r3['values'].get('R', float('nan')):.3g} is "
                        f"sharper than the start and wants **{_r3['n_phi_suggested']} φ "
                        f"samples** — this fit used {_r3['n_phi']} (held fixed to keep the "
                        "gradients clean). **Run again** from these values to refine at the "
                        "finer sampling.")
                if _r3.get("report"):
                    with st.expander(f"lmfit fit report ({_r3.get('method','')})",
                                     expanded=False):
                        st.code(_r3["report"], language="text")
                        st.download_button("Download fit report (.txt)",
                                           _r3["report"].encode("utf-8"),
                                           file_name="stage3_fit_report.txt",
                                           mime="text/plain", key="s3_dl")
                st.caption("Apply the refined values on the **Simulation** tab to carry "
                           "them into the model.")
            _vc = st.columns([2, 1, 1])
            with _vc[0]:
                _s3_pct = st.slider(
                    "Display contrast (percentile)", min_value=90.0, max_value=100.0,
                    value=float(cake_percentile), step=0.1, key="s3_pct",
                    help="Upper clip for the colour scale. Lower brightens faint rings; "
                         "higher darkens the image by clipping fewer bright pixels.")
            with _vc[1]:
                _s3_cmap = st.selectbox("Colour map", ["Inferno", "Magma", "Viridis",
                                                       "Cividis", "Hot", "Greys"],
                                        key="s3_cmap")
            with _vc[2]:
                _s3_dmap = st.selectbox("Difference map", ["RdBu", "Picnic", "BrBG",
                                                           "Spectral"], key="s3_dmap")
            st.caption("Click-drag on any panel to zoom — all three share the 2θ axis. "
                       "Double-click to reset. Fitted windows are outlined in cyan.")
            st.plotly_chart(
                cp.plot_stage3_comparison(_bl, _ev3["sim"], percentile=float(_s3_pct),
                                          colorscale=_s3_cmap, diff_colorscale=_s3_dmap),
                width='stretch')

            # --- Ring intensity vs azimuth, collapsing each 2θ window ---
            _s3_meas = st.radio(
                "Ring profile measure", ["Integrated", "Peak"], horizontal=True,
                key="s3_measure",
                help="How each ring's 2θ window is collapsed to one value per azimuth "
                     "box. Integrated = sum across the window × box width (the ring's "
                     "integrated intensity, unaffected by the peak drifting within the "
                     "window). Peak = the maximum. Comparing the two separates an "
                     "intensity error from a peak-width error.")
            _meas = "peak" if _s3_meas == "Peak" else "integrated"
            _s3_prof = cp.stage3_azimuthal_profiles(_bl, _ev3["sim"], measure=_meas)
            st.plotly_chart(
                cp.plot_stage3_azimuthal(_s3_prof, ncols=4, measure=_meas),
                width='stretch')
            st.dataframe(
                pd.DataFrame([{"hkl": _L,
                               "2θ window": "{:.3f}–{:.3f}".format(
                                   float(_bl.tth_edges[_bl.windows[_L][0]]),
                                   float(_bl.tth_edges[min(_bl.windows[_L][1],
                                                           _bl.tth_edges.size - 1)])),
                               "boxes": int(_bl.data.shape[0]
                                            * (_bl.windows[_L][1] - _bl.windows[_L][0])),
                               "RMSE": _ev3["per_hkl"].get(_L, float("nan"))}
                              for _L in _bl.labels]),
                hide_index=True, use_container_width=True,
                column_config={"RMSE": st.column_config.NumberColumn(format="%.4g")})
        else:
            st.caption("Press **Preview image model** to overlay the current model on the "
                       "blocked image, or **Run Stage 3 refinement** to fit.")
