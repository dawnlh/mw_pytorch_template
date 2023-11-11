## trainer
trainer = dict(
    name = 'trainer', # name of the trainer file in srcs/trainer
    num_epochs=20,    # total epochs to run
    resume = None,    # path of the checkpoint to resume
    resume_conf = ['epoch', 'optimizer'], # resume config
    save_latest_k=5,   # save the latest k checkpoint
    milestone_ckp = [], # save the checkpoint at the milestone epoch
    logging_interval=2, # log interval
    eval_interval=1, # eval interval
    max_iter=None # limit max_iter for each epoch
    )