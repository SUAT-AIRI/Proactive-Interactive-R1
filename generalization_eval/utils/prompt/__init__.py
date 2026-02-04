import os
import os.path as osp

current_dir = osp.dirname(__file__)


with open(osp.join(current_dir, 'user_simulator_collab_prompt.txt'), 'r') as f:
    USER_SIMULATOR_COLLAB_PROMPT = f.read()

with open(osp.join(current_dir, 'user_simulator_QA_prompt.txt'), 'r') as f:
    USER_SIMULATOR_QA_PROMPT = f.read()

with open(osp.join(current_dir, 'user_simulator_FK_prompt.txt'), 'r') as f:
    USER_SIMULATOR_FK_PROMPT = f.read()

with open(osp.join(current_dir, 'accuracy_prompt.txt'), 'r') as f:
    ACC_PROMPT = f.read()

with open(osp.join(current_dir, 'extract_match_prompt.txt'), 'r') as f:
    EXTRACT_MATCH_PROMPT = f.read()
