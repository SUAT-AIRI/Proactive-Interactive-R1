import os
import os.path as osp

current_dir = osp.dirname(__file__)


with open(osp.join(current_dir, 'user_simulator_collab_prompt.txt'), 'r') as f:
    USER_SIMULATOR_COLLAB_PROMPT = f.read()