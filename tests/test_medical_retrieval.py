import unittest

from aibondhu.medical_retrieval import MedicalEvidenceIndex

class MedicalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.records = [{
            "question": "জ্বর হলে কী করা উচিত",
            "answer": "বিশ্রাম নিন এবং পর্যাপ্ত তরল পান করুন। অবস্থা খারাপ হলে চিকিৎসকের পরামর্শ নিন।",
            "source_title": "Approved health guidance",
            "source_url": "https://example.org/health",
            "license": "CC-BY-4.0",
            "review_status": "approved",
            "reviewer_role": "licensed_clinician"
        }]

    def test_retrieves_approved_evidence(self):
        result = MedicalEvidenceIndex(self.records).answer("জ্বর হলে কী করা উচিত")
        self.assertEqual(result["status"], "retrieved_evidence")
        self.assertEqual(len(result["evidence"]), 1)

    def test_rejects_unreviewed_record(self):
        row = dict(self.records[0], review_status="pending")
        result = MedicalEvidenceIndex([row]).answer("জ্বর হলে কী করা উচিত")
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_emergency_escalation(self):
        result = MedicalEvidenceIndex(self.records).answer("আমার শ্বাসকষ্ট হচ্ছে")
        self.assertEqual(result["status"], "emergency")

    def test_unknown_question_abstains(self):
        result = MedicalEvidenceIndex(self.records).answer("কিডনি প্রতিস্থাপনের জটিল চিকিৎসা")
        self.assertEqual(result["status"], "insufficient_evidence")

if __name__ == "__main__":
    unittest.main()
