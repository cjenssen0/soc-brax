#@title All `dm_control` imports required for this tutorial

# soc algo
from soc_brax.soc import soc

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
# from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

##% Other helper functions
#@title Other imports and helper functions

# General
import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image

#@title Visualizing an initial state of one task per domain in the Control Suite
domains_tasks = {domain: task for domain, task in (suite.ALL_TASKS)}
random_state = np.random.RandomState(42)
num_domains = len(domains_tasks)
n_col = num_domains // int(np.sqrt(num_domains))
n_row = num_domains // n_col + int(0 < num_domains % n_col)
_, ax = plt.subplots(n_row, n_col, figsize=(12, 12))
for a in ax.flat:
  a.axis('off')
  a.grid(False)

print(f'Iterating over all {num_domains} domains in the Suite:')
# for j, [domain, task] in enumerate(domains_tasks.items()):
# for j, [domain, task] in enumerate({'acrobot': 'swingup_sparse'}.items()):
for j, [domain, task] in enumerate({'swimmer': 'swimmer15'}.items()):
  print(domain, task)

  env = suite.load(domain, task, task_kwargs={'random': random_state})
  timestep = env.reset()
  pixels = env.physics.render(height=200, width=200, camera_id=0)

  ax.flat[j].imshow(pixels)
  ax.flat[j].set_title(domain + ': ' + task)

plt.savefig("all_envs.png")
clear_output()

agent = soc(env)