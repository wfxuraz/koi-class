from koi.predict import format_predictions, rank_predictions


def test_rank_predictions_sorts_descending():
    classes = ["a", "b", "c"]
    probs = [0.2, 0.7, 0.1]
    ranked = rank_predictions(probs, classes, top_k=0, min_prob=0.0)
    assert ranked[0] == ("b", 0.7)
    assert [name for name, _ in ranked] == ["b", "a", "c"]


def test_rank_predictions_top_k_and_threshold():
    classes = ["a", "b", "c"]
    probs = [0.2, 0.7, 0.1]
    assert rank_predictions(probs, classes, top_k=2, min_prob=0.0) == [
        ("b", 0.7), ("a", 0.2)
    ]
    assert rank_predictions(probs, classes, top_k=0, min_prob=0.15) == [
        ("b", 0.7), ("a", 0.2)
    ]


def test_format_predictions_renders_percentages():
    ranked = [("kohaku", 0.804), ("ginrin-kohaku", 0.142)]
    text = format_predictions(ranked)
    assert "kohaku" in text
    assert "80.4%" in text
    assert "14.2%" in text
