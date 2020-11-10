from adversarial.diayn.discrete import DiscreteDiayn
from adversarial.diayn.continuous import ContinuousDiayn
from adversarial.env import make_env
from pyrl.logger import Loggable


def diayn_test(diayn):
    for dev in "cpu", "cuda":
        env = make_env("InvertedPendulum-v2", dev)
        dd = diayn(env.observation_space, [256, 256], 10, 3, 128, _device=dev)
        obs = env.reset()
        obs["diayn"] = dd.sample(1)
        print(obs)
        dd.calc_rewards(obs)
        dd.train(obs)

        assert isinstance(dd, Loggable)
        print(dd.log_hyperparams())
        print(dd.log_epoch())


def test_discrete_diayn_integration():
    diayn_test(DiscreteDiayn)


def test_continuous_diayn_integration():
    diayn_test(ContinuousDiayn)
