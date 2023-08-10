import wandb
def setWandb(args, name="ResNet"):
    # Set the wandb project where this run will be logged
        wandb.init(
        # track hyperparameters and run metadata
        project='KGFL-Res18-Knowledge',

        config={
        "learning_rate" : args.local_learning_rate,
        "architecture" : args.modelType,
        "dataset": args.dataset,
        "epochs" : args.global_rounds,
        "name" : name,
        }
        )


def finishWandb():
    wandb.finish()