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


tab_sim, tab_cake, tab_peaks = st.tabs(["Simulation", "Cake Import & Background", "Peak Extraction"])

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
                    generate_1D_XRD_overlay(XRD_df, x_exp, y_exp)
        
                #Construct the default parameter dictionary for refinement
                other = {"chi" : chi}
        
                setup_refinement_toggles(lattice_params, symmetry=symmetry, cijs=cijs, stress=sigma_params, other=other)
            
                if st.button("Refine XRD"):
                    phi_values = np.radians(np.arange(0, 360, 10))
                    psi_values = 1
                
                    result = run_refinement(st.session_state.ref_params, st.session_state.refine_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, 
                                            phi_values, psi_values, wavelength, symmetry, x_exp, y_exp, lattice_params, cijs,
                                            sigma_params, chi, Funamori_broadening, po_model=po_model)
            
                    if result.success:
                        st.success("Refinement successful!")
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
