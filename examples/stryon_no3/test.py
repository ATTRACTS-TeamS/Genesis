import pickle

with open("logs/stryon_no3_train/cfgs.pkl", "rb") as f:
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(f)

print(train_cfg["algorithm"])
