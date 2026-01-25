import pytest

from bijective_matching import bijcheck, s_score


def test_bijcheck_perfect_match():
    group_ids_1 = [0, 0, 1, 1, 2, 2]
    group_ids_2 = [0, 0, 1, 1, 2, 2]

    e_num, e_den, q_num, q_den = bijcheck(group_ids_1, group_ids_2, min_group_size=2)

    assert e_num == 3
    assert e_den == 3
    assert q_num == 6
    assert q_den == 6


def test_bijcheck_with_isolated_groups():
    group_ids_1 = [0, 0, 1]
    group_ids_2 = [-1, -1, -1]

    e_num, e_den, q_num, q_den = bijcheck(group_ids_1, group_ids_2, min_group_size=2)

    assert e_num == 0
    assert e_den == 1
    assert q_num == pytest.approx(1.0)
    assert q_den == pytest.approx(2.0)


def test_bijcheck_requires_equal_length():
    with pytest.raises(AssertionError):
        bijcheck([0, 1], [0], min_group_size=1)


def test_s_score_perfect_match():
    group_ids_1 = [0, 0, 1, 1]
    group_ids_2 = [0, 0, 1, 1]

    s_value, e_value, q_value = s_score(group_ids_1, group_ids_2, groupcut=2)

    assert s_value == pytest.approx(1.0)
    assert e_value == pytest.approx(1.0)
    assert q_value == pytest.approx(1.0)
