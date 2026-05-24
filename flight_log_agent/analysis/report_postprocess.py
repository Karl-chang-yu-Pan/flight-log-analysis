from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from flight_log_agent.audit import DeveloperAuditLogger
from flight_log_agent.models import FlightLogReport, PlotRef
from flight_log_agent.ulog.plots import generate_signal_plot


AuditCall = Callable[..., Any]


def generate_report_plots(
    report: FlightLogReport,
    log_path: Path,
    output_dir: Path,
    audit_logger: Optional[DeveloperAuditLogger],
    audit_call: Optional[AuditCall] = None,
    plot_generator: Callable[..., dict[str, Any]] = generate_signal_plot,
) -> FlightLogReport:
    for hyp in report.ranked_hypotheses:
        for plot in hyp.plots:
            if plot.start_s is None or plot.end_s is None or not plot.signals:
                continue
            args = (
                log_path,
                output_dir,
                plot.title,
                float(plot.start_s),
                float(plot.end_s),
                plot.signals,
                plot.purpose,
            )
            kwargs = {
                "plot_type": plot.plot_type,
                "bins": plot.bins,
                "overlays": [o.model_dump(exclude_none=True) for o in plot.overlays],
            }
            if audit_call is None:
                result = plot_generator(*args, **kwargs)
            else:
                result = audit_call(
                    audit_logger,
                    "postprocess_plot",
                    "generate_signal_plot",
                    plot_generator,
                    {
                        "title": plot.title,
                        "start_s": plot.start_s,
                        "end_s": plot.end_s,
                        "signals": plot.signals,
                    },
                    *args,
                    **kwargs,
                )
            apply_plot_result(plot, result)
    return report


def apply_plot_result(plot: PlotRef, result: dict[str, Any]) -> None:
    for key in ("title", "path", "purpose", "signals", "plot_type", "overlays", "missing_signals", "warnings"):
        if key in result:
            setattr(plot, key, result[key])
