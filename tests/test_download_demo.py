import pytest

from scripts.download_demo import repo_id


def test_repo_id_accepts_dataset_url_or_id():
    expected = "mikkochi/AETHER_Demo"
    assert repo_id("https://huggingface.co/datasets/mikkochi/AETHER_Demo") == expected
    assert repo_id(expected) == expected


@pytest.mark.parametrize("value", ["", "mikkochi", "a/b/c", "https://example.com/a/b"])
def test_repo_id_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        repo_id(value)
