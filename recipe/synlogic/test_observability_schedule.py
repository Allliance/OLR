import unittest
from .observability import snapshot_due
class ScheduleTests(unittest.TestCase):
    def test_periodic_eval(self):
        self.assertFalse(snapshot_due(458,20,840))
        self.assertTrue(snapshot_due(459,20,840))
        self.assertFalse(snapshot_due(460,20,840))
        self.assertTrue(snapshot_due(479,20,840))
    def test_final_eval(self):
        self.assertTrue(snapshot_due(838,20,839))
        self.assertFalse(snapshot_due(839,20,839))
        self.assertTrue(snapshot_due(839,20,840))
        self.assertFalse(snapshot_due(840,20,840))
    def test_disabled(self):
        self.assertFalse(snapshot_due(839,0,840))
if __name__=="__main__":unittest.main()
