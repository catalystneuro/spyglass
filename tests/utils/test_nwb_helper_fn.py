import datetime

import numpy as np
import pynwb
import pytest

from spyglass.utils.nwb_helper_fn import _get_epoch_groups


@pytest.fixture(scope="module")
def get_electrode_indices(common):
    from spyglass.common import get_electrode_indices  # noqa: E402

    return get_electrode_indices


@pytest.fixture(scope="module")
def custom_nwbfile(common):
    nwbfile = pynwb.NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )
    dev = nwbfile.create_device(name="device")
    elec_group = nwbfile.create_electrode_group(
        name="electrodes",
        description="description",
        location="location",
        device=dev,
    )
    for i in range(10):
        nwbfile.add_electrode(
            id=100 + i,
            x=0.0,
            y=0.0,
            z=0.0,
            imp=-1.0,
            location="location",
            filtering="filtering",
            group=elec_group,
        )
    electrode_region = nwbfile.electrodes.create_region(
        name="electrodes",
        region=[2, 3, 4, 5],
        description="description",  # indices
    )
    nwbfile.add_acquisition(
        pynwb.ecephys.ElectricalSeries(
            name="eseries",
            data=[0, 1, 2],
            timestamps=[0.0, 1.0, 2.0],
            electrodes=electrode_region,
        )
    )
    yield nwbfile


def test_electrode_nwbfile(get_electrode_indices, custom_nwbfile):
    ret = get_electrode_indices(custom_nwbfile, [102, 105])
    assert ret == [2, 5]


def test_electrical_series(get_electrode_indices, custom_nwbfile):
    eseries = custom_nwbfile.acquisition["eseries"]
    ret = get_electrode_indices(eseries, [102, 105])
    assert ret == [0, 3]


class TestGetEpochGroups:
    @pytest.fixture
    def position_with_timestamps(self):
        spatial_series = pynwb.behavior.SpatialSeries(
            name="series_0",
            data=np.zeros((100, 2)),
            timestamps=np.linspace(5.0, 10.0, 100),
            reference_frame="unknown",
        )
        return pynwb.behavior.Position(spatial_series=spatial_series)

    @pytest.fixture
    def position_with_rate(self):
        spatial_series = pynwb.behavior.SpatialSeries(
            name="series_0",
            data=np.zeros((100, 2)),
            starting_time=5.0,
            rate=30.0,
            reference_frame="unknown",
        )
        return pynwb.behavior.Position(spatial_series=spatial_series)

    def test_timestamps(self, position_with_timestamps):
        result = _get_epoch_groups(position_with_timestamps)
        assert result == {5.0: [0]}

    def test_starting_time_and_rate(self, position_with_rate):
        result = _get_epoch_groups(position_with_rate)
        assert result == {5.0: [0]}
