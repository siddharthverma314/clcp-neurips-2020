from pyrl.logger import simpleloggable


def test_loggable():
    @simpleloggable
    class Test:
        def __init__(self, nolog, _log):
            pass

        def test_log(self):
            self.log("a", 1)
            self.log("b", 2)

    a = Test("bye", "hi")
    a.test_log()
    assert a.log_hyperparams() == {"log": "hi"}
    assert a.log_epoch() == {"a": 1, "b": 2}
