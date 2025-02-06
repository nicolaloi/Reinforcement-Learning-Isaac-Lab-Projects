from pathlib import Path

EXPERIMENTAL_LEGGED_POLICY_FOLDER = Path(__file__).parent / "policy"

EXPERIMENTAL_LEGGED_STANDUP_VELOCITY_ROUGH_POLICY_PATH = \
    str(EXPERIMENTAL_LEGGED_POLICY_FOLDER / 
        "experimental_legged_standup_velocity_rough_policy_50000_250118/exported/policy.pt")