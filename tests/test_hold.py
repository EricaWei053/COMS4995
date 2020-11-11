import os.path
def test_1():
    assert 0 == 0

def test_2():
    assert 1 != 2

def test_3():
    assert os.path.exists("../data/STRG1.csv")

def test_4():
    assert os.path.exists("../data/STRG2.csv")

def test_5():
    assert os.path.exists("../src/plot.py")

