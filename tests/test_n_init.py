import numpy as np
from kluster_fudge.core import KModes


def test_n_init_basic():
    # Create simple categorical data
    data = np.array(
        [
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["x", "y", "z"],
            ["x", "y", "z"],
            ["x", "y", "z"],
        ]
    )

    # Test with n_init = 1
    km1 = KModes(n_clusters=2, n_init=1, random_state=42)
    km1.fit(data)
    # perfect clustering should be possible
    assert km1.cost_ == 0.0

    # Test with n_init = 5
    km5 = KModes(n_clusters=2, n_init=5, random_state=42)
    km5.fit(data)
    assert km5.cost_ == 0.0
