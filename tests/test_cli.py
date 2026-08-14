from argparse import Namespace

from eclipse_align.cli import resolve_preview_jobs


def test_resolve_preview_jobs_leaves_two_cores_free():
    args = Namespace(jobs=8)

    assert resolve_preview_jobs(args) == 6


def test_resolve_preview_jobs_keeps_single_worker_floor():
    args = Namespace(jobs=2)

    assert resolve_preview_jobs(args) == 1
