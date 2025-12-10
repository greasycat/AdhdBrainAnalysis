import unittest
from unittest_prettify.colorize import colorize, GREEN
from analysis.subject import SubjectInfo

@colorize(color=GREEN)
class TestSubjectInfo(unittest.TestCase):
    def test_get_rest_image(self):
        subject_info = SubjectInfo("sub-70001")
        rest_image = subject_info.get_rest_image()
        self.assertTrue(rest_image.exists())

if __name__ == "__main__":
    unittest.main()