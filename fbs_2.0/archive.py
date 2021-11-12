"""Tools for use of project-generated opsim simulations and analysis.
"""

__all__ = [
    "git_runs",
    "get_family_runs",
    "download_runs",
    "get_metric_sets",
    "get_metric_summaries",
    "normalize_metric_summaries",
    "plot_run_metric",
    "plot_run_metric_mesh",
    "get_family_descriptions",
    "describe_families",
]


import warnings
import os
import urllib
import copy

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet
import cycler
import IPython

import rubin_sim
from rubin_sim import maf

FAMILY_SOURCE = os.environ.get(
    "RUBIN_SIM_FAMILY_SOURCE",
    "https://raw.githubusercontent.com/lsst-pst/survey_strategy/main/fbs_2.0/runs_v2.0.json",
)
METRIC_SET_SOURCE = os.environ.get(
    "RUBIN_SIM_METRIC_SET_SOURCE",
    "https://raw.githubusercontent.com/ehneilsen/survey_strategy/main/fbs_2.0/metric_sets.json",
)
SUMMARY_SOURCE = os.environ.get(
    "RUBIN_SIM_SUMMARY_SOURCE",
    "https://raw.githubusercontent.com/lsst-pst/survey_strategy/main/fbs_2.0/summary_11_8.csv",
)
OPSIM_DB_DIR = os.environ.get(
    "OPSIM_DB_DIR",
    "/sims_maf/fbs_2.0"
    if os.uname().nodename.endswith(".datalab.noao.edu")
    else os.getcwd(),
)
BY_RUN_COLS = ["run", "brief", "filepath", "url"]


def get_family_runs(run_source=None):
    """Load a data frame that supplies run names for each run family

    Parameters
    ----------
    run_source : `None` or `str`
        File name or URL for the json file from which to load the metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant.

    Returns
    -------
    families : `pandas.DataFrame`
        ``families``
            The index is the run family. (`str`)
        ``run``
            the project-standard name for the run (`str`)
        ``OpsimGroup``
            The name for the group to which the runs belong (`str`)
        ``OpsimComment``
            Short description of the run (`str`)
        ``OpsimVersion``
            Opsim version name (`str`)
        ``OpsimDate``
            Date for the version of opsim (TODO: ?)
        ``brief``
            A list of descriptions for the run. Runs may have a
            different description for each family it belongs to, so it
            a list of the same length as the families column (`list`
            [`str`])
        ``url``
            The URL from which the opsim output database for this run can be
            downloaded.

    Notes
    -----
    Because runs can be members of multiple families, more than one row may
    provide metadata on the same run.

    The same content (in a different form) can be obtained using
    ``get_runs``. ``get_runs`` is more convenient when indexing by
    run; ``get_family_runs`` when indexing by family.

    """

    run_source = FAMILY_SOURCE if run_source is None else run_source
    if isinstance(run_source, pd.DataFrame):
        runs = run_source
    else:
        families = pd.read_json(run_source, orient="index")
        families.index.name = "family"
        runs = families.explode(BY_RUN_COLS)

    return runs


def get_runs(run_source=None):
    """Load metadata on opsim runs into a `pandas.DataFrame`.

    Parameters
    ----------
    run_source : `None` or `str`
        File name or URL for the json file from which to load the metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant.

    Returns
    -------
    runs : `pandas.DataFrame`
        ``run``
            The index of the DataFrame is the project-standard name for the run
            (`str`)
        ``familie``
            A list of run families to which this run belongs (`list` [`str`])
        ``version``
            The simulation version
        ``brief``
            A list of descriptions for the run. Runs may have a
            different description for each family it belongs to, so it
            a list of the same length as the families column (`list`
            [`str`])
        ``filepath``
            The file path, relative to a base opsim output directory.
        ``url``
            The URL from which the opsim output database for this run can be
            downloaded.

    Notes
    -----
    The same content (in a different form) can be obtained using
    ``get_family_runs``. ``get_runs`` is more convenient when indexing by
    run; ``get_family_runs`` when indexing by family.

    """

    family_runs = get_family_runs(run_source)

    by_family_cols = list(
        [c for c in family_runs.reset_index().columns if c not in BY_RUN_COLS]
    )

    runs = (
        family_runs.reset_index()
        .groupby(BY_RUN_COLS)
        .agg({c: list for c in by_family_cols})
        .reset_index()
        .set_index("run")
        .loc[:, ["family", "version", "brief", "filepath", "url"]]
    )

    return runs


def download_runs(runs, dest_dir=None, runs_source=None, clobber=False):
    """Download opsim visit databases for specified runs to a local directory.

    Parameters
    ----------
    runs : `pandas.DataFrame` or iterable [`str`]
        If a `pandas.DataFrame` is provided, the `OpsimRun` column will be used
        to get run names, and data will be read from the url specified in the
        `url` column.
        If a collection of `str` is provided, these will be interpreted as
        run names supplied by data originating in the run metadata provided
        by the ``runs_source`` parameter.
    dest_dir : `str`
        The local directory into which to write downloaded visit databases.
    runs_source : `str`
        File name or URL for the json file from which to load the metadata.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.RUNS_SOURCE` constant. This parameter is ignored
        if the ``runs`` parameter is set to a `pandas.DataFrame`.
    clobber : `bool`
        If ``False``, runs that would clobber an existing file will be skipped.
        If ``True``, existing files will be overwritten.


    Returns
    -------
    runs : `pandas.DataFrame`
        Metadata on runs downloaded (in the same structure as the return of
        ``archive.get_runs``).
    """

    if isinstance(runs, str):
        runs = [runs]

    if not isinstance(runs, pd.DataFrame):
        all_runs = get_runs(runs_source)
        runs = all_runs.loc[runs, :]

    if dest_dir is None:
        dest_dir = OPSIM_DB_DIR

    if not os.path.exists(dest_dir):
        raise FileNotFoundError(dest_dir)

    dest_fnames = pd.Series(
        name="fname", index=pd.Index([], name="OpsimRun"), dtype=object
    )

    for run_name, run in runs.iterrows():
        dest_fnames[run_name] = os.path.join(dest_dir, run.filepath)

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(dest_fnames[run_name]), exist_ok=True)

        if clobber or not os.path.exists(dest_fnames[run_name]):
            urllib.request.urlretrieve(run.url, dest_fnames[run_name])
        else:
            warnings.warn(
                f"{dest_fnames[run_name]} already exists; not downloading"
            )

    return dest_fnames


def get_metric_sets(metric_set_source=METRIC_SET_SOURCE):
    """Get metadata on named sets of related metrics.

    Parameters
    ----------
    metric_set_source : `str`
        File name or URL for the json file from which to load the data.
        If it is set to `None`, the data is loaded from the URL specified
        by the `archive.METRIC_SET_SOURCE` constant.

    Returns
    -------
    metric_sets : `pandas.DataFrame`
        ``metric_set``
            The 1st level of the index is the name of a set of metrics (`str`).
        ``metric``
            The 2rd level of the index is the full name of the metric (`str`).
        ``metric``
            The full name of the metric (`str`).
        ``short_name``
            An abbreviated name for the metric (`str`)..
        ``short_names_norm``
            The name for the metric after normalization (`str`).
        ``style``
            The ``matplotlib`` linestyle suggested for plots of the
            metric (`str`).
        ``invert``
            When normalizing, invert the metric value first? (`bool`)
        ``mag``
            Is the value an (astronomical) magnitude? (`bool`)
    """
    metric_set_source = (
        METRIC_SET_SOURCE if metric_set_source is None else metric_set_source
    )
    if isinstance(metric_set_source, pd.DataFrame):
        metric_sets = metric_set_source
    else:
        metric_sets = (
            pd.read_json(metric_set_source)
            .set_index("metric set")
            .set_index("metric", append=True, drop=False)
        )
    return metric_sets


def get_metric_summaries(
    run_families=tuple(),
    metric_sets=tuple(),
    runs=tuple(),
    metrics=tuple(),
    summary_source=None,
    runs_source=None,
    metric_set_source=None,
    run_order="summary",
    metric_order="summary",
):
    """Get summary metric values for a set of runs and metrics.

        Parameters
        ----------
        run_families : iterable [`str`]
            Families of runs to include in the summary.
        metric_sets : iterable [`str`]
            Sets of metrics to include in the summary.
        runs : iterable [`str`]
            Runs to include in the summary (in addition to any that are part
            of families included in ``run_families``).
        metrics : iterable [`str`]
    `       Metrics to include in the summary (in addition to any that are part of
            sets included in ``metric_sets``).
        summary_source : `str` or `pandas.DataFrame`
            File name or URL for the GNW file from which to load the data.
            If it is set to `None`, the data is loaded from the URL specified
            by the `archive.METRIC_SET_SOURCE` constant.
            If the supplied value is a `pandas.DataFrame`, it the table
            returned will be a subset of this supplied table.
        run_source : `pandas.DataFrame` or `str`
            Either a `pandas.DataFrame` of runs metadata (as returned by
            `archive.get_runs`), or a file name or URL for the json file
            from which to load the run metadata.
            If it is set to `None`, the data is loaded from the URL specified
            by the `archive.RUNS_SOURCE` constant.
        metric_set_source : `pandas.DataFrame` or `str`
            Either a `pandas.DataFrame` of metric set specifications
            (as returned by `archive.get_metric_sets`) or a
            file name or URL for the json file from which to load the data.
            If it is set to `None`, the data is loaded from the URL specified
            by the `archive.SUMMARY_SOURCE` constant.
        run_order : `str`
            Sort runs according to family definition ("family") or summary file
            ("summary") order.
        metric_order : `str`
            Sort metrics according to set definition ("set") or summary file
            ("summary") order.

        Returns
        -------
        summaries : `pandas.DataFrame`
            Metric summary values are returned in a `pandas.DataFrame`, with each
            column providing the metrics for one run, and each row the values for
            one metric. The metric names constitute the index, and the
            column names are the canonical run names.
    """
    summary_source = (
        SUMMARY_SOURCE if summary_source is None else summary_source
    )
    runs = list(runs)
    metrics = list(metrics)

    if isinstance(run_families, str):
        run_families = [run_families]

    if isinstance(metric_sets, str):
        metric_sets = [metric_sets]

    if isinstance(summary_source, pd.DataFrame):
        all_summaries = summary_source.T
    else:
        all_summaries = pd.read_csv(
            summary_source, index_col=0, low_memory=False
        )
        all_summaries.index.name = "OpsimRun"

    if len(run_families) > 0:
        families = get_family_runs(runs_source)
        for run_family in run_families:
            runs.extend(pd.Series(families.loc[run_family, "run"]).tolist())

    if len(metric_sets) > 0:
        metric_set_df = get_metric_sets(metric_set_source)
        for metric_set in metric_sets:
            metrics.extend(list(metric_set_df.loc[metric_set, "metric"]))

    if len(runs) == 0:
        runs = slice(None)
    else:
        if run_order == "summary":
            runs = [r for r in all_summaries.index if r in runs]

    if len(metrics) == 0:
        metrics = slice(None)
    else:
        #    if not isinstance(metrics, slice):
        requested_metrics = copy.copy(metrics)
        for metric in requested_metrics:
            if metric not in all_summaries.columns:
                warnings.warn(f'Metric "{metric}" not in summary, skipping')
                metrics.remove(metric)

        if metric_order == "summary":
            metrics = [m for m in all_summaries.columns if m in metrics]

    summaries = all_summaries.loc[runs, metrics].T
    summaries.index.name = "metric"
    return summaries


def normalize_metric_summaries(
    baseline_run,
    summary_source=None,
    metric_set_source=None,
):
    """Create a normalized `pandas.DataFrame` of metric summary values.

    Parameters
    ----------
    baseline_run : `str`
        The name of the run that defines a normalized value of 1.
    summary_source : `pandas.DataFrame` or `str`
        The summary metrics to normalize (as returned by `get_metric_summaries`)
        or URL or file name from which they can be loaded. (Defaults to
        ``archive.SUMMARY_SOURCE``.)
    metric_set_source : `pandas.DataFrome` or `str`
        Metric metadata as returned by `archive.get_metric_sets`, or the
        file name or URL from which they can be loaded.

    Returns
    -------
    norm_summary : `pandas.DataFrame`
        Metric summary values are returned in a `pandas.DataFrame`, with each
        column providing the metrics for one run, and each row the values for
        one metric. The metric names constitute the index, and the
        column names are the canonical run names.
        Values of 1 indicate metric values that match that of the baseline,
        differences with with indicate fractional improvement (if > 1) or
        degradation (if < 1).
    """
    summary = get_metric_summaries(summary_source=summary_source)
    metric_sets = get_metric_sets(metric_set_source)

    # Use only those metrics present both in the
    # summary and metrics sets dataframe
    used_metrics = [
        s for s in summary.index.values if s in metric_sets.metric.values
    ]
    runs = summary.columns.values
    summary = summary.loc[used_metrics, :].copy()

    # Get rid of duplicate metrics and runs
    summary = summary.groupby("metric").first().T.groupby("OpsimRun").first().T

    metric_sets = (
        metric_sets.groupby(level="metric").first().loc[used_metrics, :].copy()
    )

    norm_summary = pd.DataFrame(
        columns=summary.columns,
        index=summary.index,
        dtype="float",
    )

    assert not np.any(
        np.logical_and(metric_sets["invert"], metric_sets["mag"])
    )

    # Direct metrics are those that are neither inverted, nor compared as magnitudes
    direct = ~np.logical_or(metric_sets["invert"], metric_sets["mag"])
    norm_summary.loc[direct, :] = summary.loc[direct, :]

    norm_summary.loc[metric_sets["invert"], :] = (
        1.0 / summary.loc[metric_sets["invert"], :]
    )

    norm_summary.loc[metric_sets["mag"], :] = 1.0 + summary.loc[
        metric_sets["mag"], :
    ].subtract(summary.loc[metric_sets["mag"], baseline_run], axis="index")

    # Look a the fractional difference compared with the baseline
    norm_summary.loc[:, :] = 1 + (
        norm_summary.loc[:, :].sub(
            norm_summary.loc[:, baseline_run], axis="index"
        )
    ).div(norm_summary.loc[:, baseline_run], axis="index")

    # Set the index name
    norm_summary.index.name = "metric"

    # Make sure we return the rows and metrics in the original order
    norm_summary = norm_summary.loc[used_metrics, runs]

    return norm_summary


def plot_run_metric(
    summary,
    baseline_run=None,
    vertical_quantity="run",
    horizontal_quantity="value",
    vwidth=1,
    run_label_map=None,
    metric_label_map=None,
    metric_set_source=None,
    ax=None,
    cmap=colorcet.glasbey_hv,
    linestyles=[""],
    markers=["o"],
):
    """Plot normalized metric values as colored points on a cartesian plane.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        Values to be plotted. Should only include runs and metrics that
        should actually appear on the plot.
    baseline_run : `str`
        Name of the run to use as the baseline for normalization (see
        (archive.normalize_metric_summaries).
    vertical_quantity : {'run', 'metric', 'value'}
        Should the run, metric name, or metric value be mapped onto the y axis?
    horizontal_quantity : {'run', 'metric', 'value'}
        Should the run, metric name, or metric value be mapped onto the x axis?
    vwidth : `float`
        The width of the plot, in normalized metrics summary units. (The limits
        of the x axis will be 1-vwidth/2 and 1+width/2).
    run_label_map : mapping
        A python `mapping` between canonical run names and run labels as they
        should appear on plot labels.
    metric_label_map : mapping
        A python `mapping` between canonical metric names and metric labels
        as they should appear on plot labels.
    metric_set_source : `pandas.DataFrome` or `str`
        Metric metadata as returned by `archive.get_metric_sets`, or the
        file name or URL from which they can be loaded.
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the data.
    cmap : `matplotlib.colors.ColorMap`
        The color map to use for point colors.
    linestyles : `list`
        A list of matplotlib linestyles to use to connect the lines
    markers : `list`
        A list of matplotlib markers to use to represent the points

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure..
    ax : `matplotilb.axes.Axes`
        The plot axes.

    """

    quantities = set(["run", "metric", "value"])
    assert horizontal_quantity in quantities
    assert vertical_quantity in quantities
    assigned_quantities = set([horizontal_quantity, vertical_quantity])
    color_quantity = (quantities - assigned_quantities).pop()

    # Normalize the summary values, if a baseline was specified
    norm_summary = (
        (
            summary
            if baseline_run is None
            else normalize_metric_summaries(
                baseline_run, summary, metric_set_source=metric_set_source
            )
        )
        .stack()
        .rename("value")
        .reset_index()
        .rename(columns={"OpsimRun": "run"})
    )

    if run_label_map is not None:
        norm_summary["run"] = norm_summary["run"].map(run_label_map)

    if metric_label_map is not None:
        norm_summary["metric"] = norm_summary["metric"].map(metric_label_map)

    plot_df = pd.DataFrame(
        {
            "x": norm_summary[horizontal_quantity],
            "y": norm_summary[vertical_quantity],
        }
    )

    if color_quantity == "value":
        warnings.warn("plot_run_metric_mesh is probably a better choice")

        # turn the continuous metric value into a categorical one
        if baseline_run is None:
            # Not normalized, so we have no idea how to scale values.
            # Just let pandas create 7 bins.
            bins = 8
        else:
            bins = (-np.inf, 0, 0.8, 0.95, 1.05, 1.2, 2, np.inf)

        plot_df["color"] = pd.cut(norm_summary[color_quantity], bins)
    else:
        plot_df["color"] = norm_summary[color_quantity]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.Figure(figsize=(6, 10)) if ax is None else ax.get_figure()

    # make the linestyles and symbols list the same length as cmap
    try:
        num_colors = len(cmap)
        colors = cmap
    except TypeError:
        num_colors = len(cmap.colors)
        colors = cmap.colors

    ls_grow = int(np.ceil(num_colors / len(linestyles)))
    linestyles = (linestyles * ls_grow)[:num_colors]
    marker_grow = int(np.ceil(num_colors / len(markers)))
    markers = (markers * marker_grow)[:num_colors]
    ax.set_prop_cycle(
        cycler.cycler(color=colors)
        + cycler.cycler(linestyle=linestyles)
        + cycler.cycler(marker=markers)
    )

    plot_df.set_index("color", inplace=True)
    for idx in plot_df.index.unique():
        ax.plot(
            plot_df.loc[idx, "x"],
            plot_df.loc[idx, "y"],
            label=str(idx).strip(),
        )

    if horizontal_quantity in ["run", "metric"]:
        ax.tick_params("x", labelrotation=90)

    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.grid()

    return fig, ax


def plot_run_metric_mesh(
    summary,
    baseline_run=None,
    color_range=1,
    run_label_map=None,
    metric_label_map=None,
    ax=None,
):
    """Plot normalized metric values as colored points on a cartesian plane.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        Values to be plotted. Should only include runs and metrics that
        should actually appear on the plot.
    baseline_run : `str`
        Name of the run to use as the baseline for normalization (see
        (archive.normalize_metric_summaries).
    color_range : `float`
        The color range of the plot, in normalized metrics summary
        units. (The color range will be 1-color_range/2 and
        1+color_range/2).
    run_label_map : mapping
        A python `mapping` between canonical run names and run labels as they
        should appear on plot labels.
    metric_label_map : mapping
        A python `mapping` between canonical metric names and metric labels
        as they should appear on plot labels.
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the data.
    cmap : `matplotlib.colors.ColorMap`
        The color map to use for point colors.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The plot figure..
    ax : `matplotilb.axes.Axes`
        The plot axes.

    """
    # Normalize the summary values, if a baseline was specified
    norm_summary = (
        summary
        if baseline_run is None
        else normalize_metric_summaries(baseline_run, summary)
    )

    vmin = 1 - color_range / 2
    vmax = vmin + color_range
    norm_values = norm_summary.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.Figure(figsize=(6, 10)) if ax is None else ax.get_figure()

    y_nums, x_nums = np.arange(norm_summary.shape[0] + 1), np.arange(
        norm_summary.shape[1] + 1
    )
    im = ax.pcolormesh(
        x_nums,
        y_nums,
        norm_values,
        vmin=vmin,
        vmax=vmax,
        cmap=colorcet.cm.coolwarm_r,
    )

    ax.set_yticks(np.arange(0.5, norm_summary.shape[0] + 0.5))
    metrics = norm_summary.index.tolist()
    if metric_label_map is None:
        metric_labels = metrics
    else:
        metric_labels = [metric_label_map[m] for m in metrics]
    ax.set_yticklabels(metric_labels)

    ax.set_xticks(np.arange(0.5, norm_summary.shape[1] + 0.5))
    runs = norm_summary.columns.tolist()
    if run_label_map is None:
        run_labels = runs
    else:
        run_labels = [run_label_map[r] for r in runs]
    ax.set_xticklabels(run_labels, rotation="vertical")

    fig.colorbar(im, ax=ax, label="Fractional difference")

    return fig, ax


def get_family_descriptions(family_source=None):
    """Get description of families or funs.

    Parameters
    ----------
    family_source : `str
        File name or URL for the json file from which to load the
        family descriptinos.  If it is set to `None`, the data is
        loaded from the URL specified by the
        `archive.FAMILY_SOURCE` constant.


    Returns
    -------
    families : `pandas.DataFrame`
        Family descriptions, with comments.

    """
    family_source = FAMILY_SOURCE if family_source is None else family_source
    if isinstance(family_source, pd.DataFrame):
        families = family_source
    else:
        families = pd.read_json(family_source, orient="index")
        families.index.name = "family"
        by_family_cols = list(
            [c for c in families.columns if c not in BY_RUN_COLS]
        )
        families = families.loc[:, by_family_cols + BY_RUN_COLS]
    return families


def describe_families(family_source=None, families=slice(None)):
    """Display (in a jupyter on IPython notebook) family descirptions

    Parameters
    ----------
    family_source : `str`
        File name or URL for the json file from which to load the
        family descriptinos.  If it is set to `None`, the data is
        loaded from the URL specified by the
        `archive.FAMILY_SOURCE` constant.
    families : iterable ['str']
        A list of family name for which to display descriptions.

    """
    family_source = FAMILY_SOURCE if family_source is None else family_source
    family_df = get_family_descriptions(family_source).loc[families, :]

    family_runs = family_df.explode(["run", "brief", "filepath"]).loc[
        :, ["run", "brief", "filepath"]
    ]

    # If there is just one run in the family, we might
    # get a pd.Series back rather than a pd.DataFrame.
    # Make sure we have a DataFrame
    if isinstance(family_runs, pd.Series):
        family_runs = pd.DataFrame([family_runs])

    for family_name, family in family_df.iterrows():
        # Use awkward appending of each line to string rather
        # than a tripple quote to keep flake8 from complaining
        # about blanks at the end of lines, which are
        # meaningful in markdown (and desired here).
        description = "---\n"
        description += f"{family.description}  \n"
        description += f"**version**: {family.version}  \n"
        description += "**runs**:  \n"

        IPython.display.display_markdown(description, raw=True)
        IPython.display.display(
            family_runs.loc[[family_name], :].set_index("brief")
        )
