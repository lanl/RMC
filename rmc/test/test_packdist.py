import pytest

from rmc.utils.packed_distributions import BasePackedDistribution


def test_base_packed_dist_exception():
    with pytest.raises(TypeError):
        BasePackedDistribution()


# def test_base_packed_dist_log_pdf_exception():
#    class TestPDist(BasePackedDistribution):

#    with pytest.raises(TypeError):
#        BasePackedDistribution()
