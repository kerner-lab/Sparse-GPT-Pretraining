import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--action", type=str, required=True)
args, remaining_args = parser.parse_known_args()

if args.action == "training":
    from training.launch_training import launch_training
    launch_training(remaining_args)
elif args.action == "post_training_evaluation":
    pass
else:
    raise Exception("Unexpected args.action")
