"""
With this script, you can extract RSTs directly from wandb.
You may need to install wandb first and adjust the WANDB_PROJECT variable below.
"""
import matplotlib.pyplot as plt
import wandb

WANDB_PROJECT = ""  # add project name here

api = wandb.Api(api_key="")
runs = api.runs(path=WANDB_PROJECT)


# %%
times = {}
for run in runs:
    bs = run.config["train"]["batch_size"]
    steps = [
        850,
        2000,
        2800,
        3500,
        4000,
        4400,
        4800,
        5100,
        5450,
        5700,
        6000,
        6200,
        6400,
        6650,
        6800,
        7100,
    ]
    df = run.history(keys=["train_time"])
    df = df.set_index("_step")
    values = df.loc[steps].values.reshape(-1).tolist()
    val_strs = [f"{x:.4f}" for x in values]
    print(f"{bs}: [{', '.join(val_strs)},],")
    times[bs] = values

# As a test, plot the times to check it looks similar to the wandb plot.
for bs, times in times.items():
    xs = []
    ys = []
    for i, time in enumerate(times):
        xs.append(i + 0.5)
        xs.append(i + 1.5)
        ys.append(time)
        ys.append(time)
    print("", bs, len(times))
    plt.plot(xs, ys, label=f"bs{bs}")
plt.legend()
plt.show()
