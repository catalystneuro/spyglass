"""Test that cameras written as plain core Devices are ingested.

Core NWB has no camera type, so a file can record a video with a plain
``Device``, next to typed cameras (here an ndx-pose ``CalibratedCamera``). The
file's task table names both cameras by ID. It verifies that:

- a plain Device that an ImageSeries records with is ingested in CameraDevice;
- a plain Device that records no video is not;
- the task's camera IDs resolve to both cameras in TaskEpoch;
- both videos are ingested in VideoFile.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from hdmf.common import DynamicTable, VectorData, VectorIndex
from ndx_pose import CalibratedCamera
from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import Device
from pynwb.file import Subject
from pynwb.image import ImageSeries

N_FRAMES = 50
RATE = 50.0
CALIBRATED_CAMERA = "DeviceTestCamera1"
PLAIN_CAMERA = "DeviceTestCamera2"
NOT_A_CAMERA = "DeviceTestStimulator3"


@pytest.fixture(scope="module")
def plain_camera_nwb(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwbfile = NWBFile(
        session_description="test plain-Device cameras",
        identifier="test_plain_device_cameras",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.subject = Subject(
        subject_id="test_plain_camera_subject", species="Rattus norvegicus"
    )

    cameras = [
        nwbfile.add_device(
            CalibratedCamera(
                name=CALIBRATED_CAMERA,
                intrinsic_matrix=np.eye(3),
                rotation_matrix=np.eye(3),
                translation_vector=np.zeros(3),
                distortion_coefficients=np.zeros(4),
            )
        ),
        nwbfile.add_device(
            Device(name=PLAIN_CAMERA, description="camera without calibration")
        ),
    ]
    nwbfile.add_device(Device(name=NOT_A_CAMERA, description="no video"))
    for camera in cameras:
        nwbfile.add_acquisition(
            ImageSeries(
                name=f"Video{camera.name}",
                description=f"video of {camera.name}",
                unit="n.a.",
                format="external",
                external_file=[f"{camera.name}.mp4"],
                starting_frame=[0],
                rate=RATE,
                num_samples=N_FRAMES,
                device=camera,
            )
        )

    nwbfile.add_epoch(
        start_time=0.0, stop_time=(N_FRAMES - 1) / RATE, tags=["01_session"]
    )
    camera_id = VectorData(
        name="camera_id", description="Camera IDs.", data=[1, 2]
    )
    task_epochs = VectorData(
        name="task_epochs", description="Task epochs.", data=[1]
    )
    task_table = DynamicTable(
        name="task_table",
        description="task of the session",
        columns=[
            VectorData(
                name="task_name",
                description="Name of the task.",
                data=["camera_device_test_task"],
            ),
            VectorData(
                name="task_description",
                description="Description of the task.",
                data=["task recorded by a typed and a plain-Device camera"],
            ),
            camera_id,
            VectorIndex(name="camera_id_index", data=[2], target=camera_id),
            task_epochs,
            VectorIndex(name="task_epochs_index", data=[1], target=task_epochs),
        ],
    )
    nwbfile.create_processing_module("tasks", "tasks").add(task_table)

    raw_file_name = "test_camera_devices.nwb"
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


def test_plain_device_camera_ingested(common, plain_camera_nwb):
    cameras = (
        common.CameraDevice
        & [{"camera_name": n} for n in (CALIBRATED_CAMERA, PLAIN_CAMERA)]
    ).fetch("camera_name", "camera_id", as_dict=True, order_by="camera_name")
    assert [(c["camera_name"], c["camera_id"]) for c in cameras] == [
        (CALIBRATED_CAMERA, 1),
        (PLAIN_CAMERA, 2),
    ]
    assert not (common.CameraDevice & {"camera_name": NOT_A_CAMERA})


def test_task_epoch_resolves_plain_device_camera(common, plain_camera_nwb):
    epoch = (common.TaskEpoch & plain_camera_nwb).fetch1()
    assert [c["camera_name"] for c in epoch["camera_names"]] == [
        CALIBRATED_CAMERA,
        PLAIN_CAMERA,
    ]


def test_plain_device_video_ingested(common, plain_camera_nwb):
    videos = (common.VideoFile & plain_camera_nwb).fetch(
        "camera_name", order_by="video_file_num"
    )
    assert sorted(videos) == [CALIBRATED_CAMERA, PLAIN_CAMERA]
