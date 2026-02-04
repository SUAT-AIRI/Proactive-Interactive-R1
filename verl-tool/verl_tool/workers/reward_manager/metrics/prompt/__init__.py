import os
import os.path as osp

current_dir = osp.dirname(__file__)

with open(osp.join(current_dir, 'helpfulness_reward_prompt.txt'), 'r') as f:
    HELPFULNESS_REWARD_PROMPT = f.read()
