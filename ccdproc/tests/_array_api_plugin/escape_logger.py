# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The escape logger: catch silent conversion of foreign array-API arrays to
NumPy.

On CuPy such a conversion typically raises immediately, because the array
lives on a GPU, which is how those bugs are usually found. On backends such
as dask and jax the conversion succeeds silently, so the bug is easy to miss.
`EscapeLog` monkeypatches ``numpy.asarray``, ``numpy.asanyarray`` and
``numpy.ma.asanyarray`` for the duration of the session so that every call
handed an array whose array-API namespace is not NumPy is logged and tallied.

Notes
-----
Only Python-level calls through those three module attributes are visible.
C-level coercions inside compiled dependencies, coercions inside third-party
libraries, and references bound before the patch (``from numpy import
asarray``) bypass the wrappers entirely, so the absence of a warning is not
proof that no conversion happened. The logger is a sampler, not a complete
net; the same blind spot carries into the tally and the baseline ratchet.

Activated by setting ``<PREFIX>_LOG_ARRAY_ESCAPES`` to a truthy value.
"""

import logging
import threading
import traceback
from collections import defaultdict

import array_api_compat
import array_api_compat.numpy
import numpy as np

from .triage import UNKNOWN_LOCATION

#: Arrays built through plain NumPy or through array-api-compat's NumPy
#: wrapper report one of these namespaces. Neither counts as an escape: we
#: only care about arrays from a genuinely different library (jax, dask,
#: cupy, array_api_strict, ...) ending up in a NumPy-only call.
NUMPY_LIKE_NAMESPACES = {np, array_api_compat.numpy}

#: The NumPy entry points the logger wraps, as {attribute owner, name}.
PATCHED_ENTRY_POINTS = (
    (np, "asarray", "numpy.asarray"),
    (np, "asanyarray", "numpy.asanyarray"),
    (np.ma, "asanyarray", "numpy.ma.asanyarray"),
)


def foreign_namespace(obj):
    """
    Return ``obj``'s array-API namespace if it is foreign, else None.

    Notes
    -----
    NumPy arrays, including ``numpy.ma`` masked arrays, are never foreign.
    Detection goes through ``array_api_compat.array_namespace()`` rather than
    the raw ``__array_namespace__`` dunder because some backends (notably
    dask) do not define the dunder on their array objects even though
    array-api-compat can resolve a namespace for them. The namespace is
    foreign unless it is NumPy or array-api-compat's NumPy wrapper -- i.e.
    exactly the arrays whose conversion to NumPy would fail (CuPy on a GPU)
    or silently densify or transfer (dask, jax). The ``try``/``except``
    guards against non-arrays and misbehaving objects, including unhashable
    namespaces; those are treated as not foreign so the logger never breaks
    the call it wraps.
    """
    if isinstance(obj, np.ndarray):
        return None
    try:
        namespace = array_api_compat.array_namespace(obj)
        if namespace in NUMPY_LIKE_NAMESPACES:
            return None
    except Exception:
        return None
    return namespace


class ReentrancyGuard(threading.local):
    """Thread-local flag keeping the wrappers from recursing into themselves."""

    def __init__(self):
        self.active = False


class EscapeLog:
    """
    Wrap NumPy's coercion entry points and tally the escapes they see.

    Parameters
    ----------
    settings : `.config.Settings`
        Supplies the activation flag and the logger name.
    classifier : `.triage.FrameClassifier`
        Used to blame the innermost library frame for each escape.

    Notes
    -----
    The tally is the observed-set input to the baseline ratchet and to the
    end-of-session summary. Unlike triaged failures these escapes do not fail
    a test -- on dask and jax the conversion succeeds -- so the tally is the
    only way to collapse the streamed warnings into something readable.
    """

    def __init__(self, settings, classifier):
        self.settings = settings
        self.classifier = classifier
        self.logger = logging.getLogger(settings.logger_name)
        #: Maps a (relfile, lineno, function, coercion) key to the number of
        #: times that coercion was logged during the session.
        self.counts = defaultdict(int)
        self.counts_lock = threading.Lock()

    @property
    def active(self):
        """True when the escape logger was requested for this session."""
        return self.settings.log_escapes

    def record(self, frame, funcname):
        """
        Tally one logged escape.

        Parameters
        ----------
        frame : traceback.FrameSummary or None
            The frame chosen by
            `.triage.FrameClassifier.locate_escape_site`.
        funcname : str
            Dotted name of the NumPy entry point that did the coercion.
        """
        if frame is None:
            key = (UNKNOWN_LOCATION, 0, "", funcname)
        else:
            key = (
                self.classifier.relpath(frame.filename),
                frame.lineno,
                frame.name,
                funcname,
            )
        with self.counts_lock:
            self.counts[key] += 1

    def observed_library_sites(self):
        """Set of (relfile, function, coercion) for the library escapes seen."""
        return {
            (relfile, function, funcname)
            for (relfile, _lineno, function, funcname) in self.counts
            if self.classifier.is_library_site(relfile)
        }

    def locate_site(self):
        """Innermost library frame of the live stack, or None."""
        return self.classifier.locate_escape_site(traceback.extract_stack())

    def make_wrapper(self, original, funcname, guard):
        """
        Build the replacement for one NumPy coercion entry point.

        Parameters
        ----------
        original : callable
            The real ``numpy.asarray``, ``numpy.asanyarray`` or
            ``numpy.ma.asanyarray``.
        funcname : str
            Dotted NumPy name recorded with each escape.
        guard : `ReentrancyGuard`
            Breaks recursion: namespace detection and stack extraction can
            themselves call the patched functions.

        Notes
        -----
        The wrapper logs and tallies an escape whenever its first positional
        argument is an array from a non-NumPy array-API library, then always
        calls ``original``. Behavior is unchanged; escapes are only observed.
        """

        def wrapper(*args, **kwargs):
            # Do nothing extra on re-entrant calls (guard held) or when
            # there is no positional argument to inspect.
            if not guard.active and args:
                guard.active = True
                try:
                    obj = args[0]
                    namespace = foreign_namespace(obj)
                    if namespace is not None:
                        # A foreign array is about to be coerced to NumPy:
                        # warn with the innermost library frame to blame, and
                        # tally the site for the end-of-session summary and
                        # the baseline ratchet.
                        frame = self.locate_site()
                        self.logger.warning(
                            "array-API escape: %s() called on a %r array "
                            "(namespace=%r) at %s",
                            funcname,
                            type(obj),
                            namespace,
                            self.classifier.describe(frame),
                        )
                        self.record(frame, funcname)
                finally:
                    guard.active = False
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", funcname)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        return wrapper

    def patch(self):
        """
        Install the wrappers.

        Returns
        -------
        callable
            A zero-argument function that restores the originals.
        """
        guard = ReentrancyGuard()
        originals = [
            (owner, attr, getattr(owner, attr))
            for owner, attr, _ in PATCHED_ENTRY_POINTS
        ]
        for owner, attr, funcname in PATCHED_ENTRY_POINTS:
            original = getattr(owner, attr)
            setattr(owner, attr, self.make_wrapper(original, funcname, guard))

        def restore():
            for owner, attr, original in originals:
                setattr(owner, attr, original)

        return restore

    def report(self, terminalreporter):
        """
        Print the live escape-log summary.

        Notes
        -----
        Collapses the streamed per-call warnings into one deduplicated,
        most-frequent-first list of call sites still to migrate.
        """
        if not self.counts:
            return

        terminalreporter.section(self.settings.section("array-API escape log summary"))
        total = sum(self.counts.values())
        terminalreporter.write_line(
            f"{total} silent numpy coercion(s) of foreign arrays across "
            f"{len(self.counts)} call site(s), most frequent first:"
        )

        ordered = sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)
        for (relfile, lineno, function, funcname), count in ordered:
            terminalreporter.write_line(
                f"    {count:>5}x  {funcname}()  {relfile}:{lineno} {function}"
            )
