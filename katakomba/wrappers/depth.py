import nle
from katakomba.env import OfflineNetHackChallengeWrapper


class DepthInfoWrapper(OfflineNetHackChallengeWrapper):
    """
    Wrapper that adds current depth to the info during step and reset.
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["current_depth"] = self.get_current_depth()
        return obs, reward, done, info
