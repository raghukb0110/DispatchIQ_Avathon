import pytest

from app.ml.dataset import build_dataset, random_scenarios
from app.ml.train import fit
from app.models import Weights


@pytest.fixture
def weights():
    return Weights()


@pytest.fixture(scope="session")
def small_scorer():
    """Quick model trained in-test so ML tests don't depend on the saved artifact."""
    ds = build_dataset(random_scenarios(20, base_seed=500), Weights())
    return fit("lgbm_ranker", ds.X, ds.y, ds.decision_ids)
