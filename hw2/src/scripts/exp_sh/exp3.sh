uv run ../run.py --env_name LunarLander-v2 \
    --ep_len 1000 --discount 0.99 -n 200 \
    -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 \
    --use_reward_to_go --use_baseline \
    --gae_lambda 0.9 \
    --exp_name lunar_lander_lambda_0.9