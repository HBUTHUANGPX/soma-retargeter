from pathlib import Path
import unittest
from unittest import mock

from app import bvh_to_csv_converter


class _FakePool:
    def __init__(self, processes, record):
        self.processes = processes
        self.record = record

    def __enter__(self):
        self.record["entered"] = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.record["exited"] = True
        return False

    def map(self, func, items):
        items = list(items)
        self.record["map_items"] = items
        return [func(item) for item in items]


class _FakeContext:
    def __init__(self, record):
        self.record = record

    def Pool(self, processes):
        self.record["processes"] = processes
        return _FakePool(processes, self.record)


class LoadBvhAnimationsBatchTests(unittest.TestCase):
    def test_rebinds_loaded_animation_to_main_process_skeleton(self):
        batch = [Path("a.bvh")]
        foreign_skeleton = object()
        main_skeleton = object()
        fake_animation = mock.Mock()
        fake_animation.skeleton = foreign_skeleton
        rebound_animation = mock.Mock()

        with mock.patch.object(bvh_to_csv_converter.os, "cpu_count", return_value=2), \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "_load_bvh_animation_task",
                 return_value=fake_animation,
             ), \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "create_animation_buffer_for_skeleton",
                 return_value=rebound_animation,
             ) as rebinder:
            animations = bvh_to_csv_converter._load_bvh_animations_batch(
                batch,
                bvh_skeleton=main_skeleton,
                expected_num_joints=29,
            )

        self.assertEqual(animations, [rebound_animation])
        rebinder.assert_called_once_with(fake_animation, main_skeleton)

    def test_defaults_to_half_cpu_count_with_spawn_pool(self):
        batch = [Path("a.bvh"), Path("b.bvh"), Path("c.bvh"), Path("d.bvh"), Path("e.bvh")]
        record = {}

        with mock.patch.object(bvh_to_csv_converter.os, "cpu_count", return_value=8), \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "_load_bvh_animation_task",
                 side_effect=lambda task: f"anim:{task[0].name}",
             ) as worker, \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "create_animation_buffer_for_skeleton",
                 side_effect=lambda animation, skeleton: animation,
             ), \
             mock.patch.object(
                 bvh_to_csv_converter.mp,
                 "get_context",
                 return_value=_FakeContext(record),
             ) as get_context:
            animations = bvh_to_csv_converter._load_bvh_animations_batch(
                batch,
                bvh_skeleton="skeleton",
                expected_num_joints=29,
            )

        self.assertEqual(record["processes"], 4)
        self.assertEqual(animations, ["anim:a.bvh", "anim:b.bvh", "anim:c.bvh", "anim:d.bvh", "anim:e.bvh"])
        self.assertEqual(len(record["map_items"]), len(batch))
        get_context.assert_called_once_with("spawn")
        self.assertEqual(worker.call_count, len(batch))

    def test_caps_process_count_to_batch_size(self):
        batch = [Path("a.bvh"), Path("b.bvh")]
        record = {}

        with mock.patch.object(bvh_to_csv_converter.os, "cpu_count", return_value=16), \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "_load_bvh_animation_task",
                 side_effect=lambda task: f"anim:{task[0].name}",
             ), \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "create_animation_buffer_for_skeleton",
                 side_effect=lambda animation, skeleton: animation,
             ), \
             mock.patch.object(
                 bvh_to_csv_converter.mp,
                 "get_context",
                 return_value=_FakeContext(record),
             ):
            bvh_to_csv_converter._load_bvh_animations_batch(
                batch,
                bvh_skeleton="skeleton",
                expected_num_joints=29,
            )

        self.assertEqual(record["processes"], 2)

    def test_uses_serial_path_when_only_one_worker_needed(self):
        batch = [Path("only.bvh")]

        with mock.patch.object(bvh_to_csv_converter.os, "cpu_count", return_value=2), \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "_load_bvh_animation_task",
                 side_effect=lambda task: f"anim:{task[0].name}",
             ) as worker, \
             mock.patch.object(
                 bvh_to_csv_converter,
                 "create_animation_buffer_for_skeleton",
                 side_effect=lambda animation, skeleton: animation,
             ), \
             mock.patch.object(bvh_to_csv_converter.mp, "get_context") as get_context:
            animations = bvh_to_csv_converter._load_bvh_animations_batch(
                batch,
                bvh_skeleton="skeleton",
                expected_num_joints=29,
            )

        self.assertEqual(animations, ["anim:only.bvh"])
        get_context.assert_not_called()
        worker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
