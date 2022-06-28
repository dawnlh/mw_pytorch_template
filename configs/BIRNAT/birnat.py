_base_=[
        "../_base_/six_gray_sim_data.py",
        # "../_base_/davis.py",
        "../_base_/birnat_davis.py",
        "../_base_/default_runtime.py"
        ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

model = dict(
    type='BIRNAT',
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/birnat/birnat.pth"