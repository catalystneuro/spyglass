"""Test ingestion of NWB core ``EventsTable`` (NWB >= 2.10, pynwb >= 4).

A minimal NWB file with one ``EventsTable`` of three events. It verifies that
the table is ingested as one ``ImportedEvents`` entry and that
``fetch1_dataframe()`` returns its events.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

pynwb_event = pytest.importorskip("pynwb.event")
if not hasattr(pynwb_event, "EventsTable"):
    pytest.skip("pynwb < 4 has no EventsTable", allow_module_level=True)

EVENT_TIMES = (0.1, 0.4, 0.7)


@pytest.fixture(scope="module")
def events_nwb(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwbfile = NWBFile(
        session_description="test imported events",
        identifier="test_imported_events",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.subject = Subject(
        subject_id="test_events_subject", species="Rattus norvegicus"
    )
    events = pynwb_event.EventsTable(
        name="Contacts", description="test contacts"
    )
    for t in EVENT_TIMES:
        events.add_row(timestamp=t)
    nwbfile.add_events_table(events)

    raw_file_name = "test_imported_events.nwb"
    nwb_path = Path(raw_dir) / raw_file_name
    key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
    if (Nwbfile & key) or nwb_path.exists():
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    insert_sessions(raw_file_name, raise_err=True)

    yield key

    with verbose_context:
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)


def test_imported_events(common, events_nwb):
    events = common.ImportedEvents & events_nwb
    entry = events.fetch1()
    assert entry["events_name"] == "Contacts"
    assert entry["description"] == "test contacts"
    assert entry["n_events"] == len(EVENT_TIMES)
    np.testing.assert_allclose(
        events.fetch1_dataframe()["timestamp"], EVENT_TIMES
    )
