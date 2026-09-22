from faceid_bench.env import library_versions


def test_missing_library_reports_none():
    versions = library_versions(("numpy", "not_a_real_library_xyz"))
    assert versions["numpy"] is not None
    assert versions["not_a_real_library_xyz"] is None
