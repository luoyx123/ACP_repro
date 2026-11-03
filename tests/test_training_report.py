"""Recorded-trajectory diagnostics must not count a successful tail as a full clip."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from acp.training.report import trajectory_metrics


class TrainingReportTest(unittest.TestCase):
    def test_tail_completion_and_early_root_failure_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rollout.npz'
            states=np.zeros((91,36));reference=np.zeros((91,26))
            np.savez(path,states=states,reference=reference,dt=.02)
            self.assertTrue(trajectory_metrics(path,20.1,.9)['reached_end'])
            self.assertFalse(trajectory_metrics(path,20.1,0.)['reached_end'])
            states[-1,0]=.777
            np.savez(path,states=states,reference=reference,dt=.02)
            result=trajectory_metrics(path,20.1,0.)
            self.assertTrue(result['root_threshold_crossed'])
            self.assertFalse(result['reached_end'])
            self.assertAlmostEqual(result['root_final_m'],.777)


if __name__=='__main__':unittest.main()
