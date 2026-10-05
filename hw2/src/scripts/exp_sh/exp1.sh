uv run ../run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole

uv run ../run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg

uv run ../run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na

uv run ../run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

uv run ../run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb

uv run ../run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg

uv run ../run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na

uv run ../run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na