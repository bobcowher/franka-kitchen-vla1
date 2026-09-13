---
title: "The environment"
part: "Part II · The pieces"
chapter: 3
weight: 3
standfirst: "What one observation is, three wrappers that make the arm behave, and the two lines the VLA changes."
---

Almost everything in this chapter arrived from the behavior-cloning project
unchanged, so we're going to read it rather than write it. It's here because you
can't reason about the readout in Chapter 7 without knowing exactly what one
observation contains, and because one of these wrappers is where the VLA's only
environment change lands.

## What the environment gives you

Franka Kitchen is a MuJoCo simulation of a 7-joint Franka arm with a two-finger
gripper, standing in a kitchen with a microwave, a kettle, two cabinets, a light
switch and four burner knobs. Let's make one and look at what comes back:

```python
env = gym.make("FrankaKitchen-v1", max_episode_steps=400,
               tasks_to_complete=["hinge cabinet"], render_mode="rgb_array")
obs, _ = env.reset()
print("keys              :", list(obs.keys()))
print("observation shape :", obs["observation"].shape)
print("action space      :", env.action_space)
```

<div class="output"><p class="output-label">This prints</p>

```text
keys              : ['observation', 'achieved_goal', 'desired_goal']
observation shape : (59,)
action space      : Box(-1.0, 1.0, (9,), float64)
```
</div>

That 59-element vector mixes joint state with the poses of every object in the
kitchen, and the two goal entries describe what counts as success. A VLA reads
the scene from pixels and the goal from language, so almost none of it is useful
to us, and Chapter 2 already explained why the `[−1, 1]` action space means we
add no normalization.

One detail in those goal vectors matters later: their width changes with
the task, giving observations of 61, 63 or 73 elements depending on what you
asked for. That was the only thing stopping demonstrations of different tasks
being mixed in one buffer, which matters in the next chapter.

## Wrapper one: stop the arm sagging

The first problem has nothing to do with learning. Send zero action for two
hundred steps and the arm droops about 61 degrees.

The cause is a feedback loop. The environment integrates your action onto the
*measured* joint position each step, gravity pulls each joint slightly below the
setpoint it was given, that droop gets read back in, and the next setpoint starts
from the drooped position. The arm ratchets downward a fraction of a degree at a
time.

<p class="listing">Listing 3.1 <em>Holding the commanded setpoint instead of the measured position</em></p>
<p class="filename">Filename: <strong>gym_robotics_custom.py</strong></p>

```python
class HeldSetpointWrapper(Wrapper):
    """Stops the arm from sagging under gravity."""

    def __init__(self, env, max_lead=0.05):
        super().__init__(env)
        self.max_lead = max_lead

    def step(self, action):
        result = self.env.step(action)
        env = self.unwrapped
        # data.ctrl is the setpoint the env just commanded, post-clipping.
        qpos = env.data.qpos[:9]
        setpoint = np.clip(env.data.ctrl[:9],
                           qpos - self.max_lead, qpos + self.max_lead)
        env.robot_env._last_robot_qpos = setpoint.copy()
        return result
```

Holding the setpoint we actually commanded turns 61 degrees of drift into about
one degree of offset, which is what a position servo is supposed to do.

`max_lead` caps how far the setpoint may run ahead of where the arm really is.
Without it the setpoint outruns the joints during a fast move and they keep
coasting for about twenty steps after you let go of the stick. It has to stay
above the servo's own steady-state droop of roughly 0.02 radians, or the clamp
re-anchors to the measured position and the sag comes straight back. At 0.05 we
measured 1.4 degrees of sag and 2.9 degrees of coast, which is the best
compromise we found.

## Wrapper two: build the observation we want

<p class="listing">Listing 3.2 <em>A camera view plus proprioception</em></p>
<p class="filename">Filename: <strong>gym_robotics_custom.py</strong></p>

```python
class VLAObservationWrapper(ObservationWrapper):
    """A camera view plus proprioception, for a VLA policy."""

    def observation(self, observation):
        robot_obs = observation["observation"]
        return {
            "camera_scene": self._grab(),
            "joint_pos": robot_obs[:9].astype(np.float32),
            "joint_vel": robot_obs[9:18].astype(np.float32),
        }
```

Three keys, and the object poses and goal vectors are gone. The camera is the
environment's free camera, which is the same view a human demonstrator sees, so
the policy and the person who taught it are looking at the same picture.

<div class="note">
<span class="note-label">Note</span>
<code>joint_pos</code> and <code>joint_vel</code> travel through the observation
but are <strong>never fed to the model</strong> in this build. That omission is
deliberate and Chapter 11 explains what it buys us, with Chapter 15 covering
when to reverse it.
</div>

## Wrapper three: match the training resolution

Frames are archived at 896 pixels so that recorded demonstrations stay useful to
a model that wants more pixels than ours does. Training reduces them to 448 as
they load, and the live environment has to apply exactly the same reduction, or
a rollout sees something subtly different from what training saw.

<p class="listing">Listing 3.3 <em>Reducing a live observation the way the loader reduces a shard</em></p>
<p class="filename">Filename: <strong>gym_robotics_custom.py</strong></p>

```python
class ObsReshapeWrapper(ObservationWrapper):
    """Brings a rendered observation down to the size the policy trained on."""

    def __init__(self, env, image_size):
        super().__init__(env)
        self.image_size = image_size
        # Fail here rather than on the first step of a rollout.
        frames.validate(env.observation_space["camera_scene"].shape[0], image_size)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "camera_scene": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
        })

    def observation(self, observation):
        return {
            **observation,
            # HWC, matching Dataset.sample_batch, so the policy sees one axis
            # order on both sides of the train/eval line.
            "camera_scene": frames.resize(observation["camera_scene"],
                                          self.image_size),
        }
```

## Putting them together

Order matters here. `ObsReshapeWrapper` goes outermost so that it reshapes
whatever the layer beneath it produced, which means anything wanting
full-resolution frames, like recording or debugging, sits below it.

<p class="filename">Filename: <strong>agent.py</strong></p>

```python
def _make_env(self, task, render_mode):
    env = gym.make("FrankaKitchen-v1", max_episode_steps=self.max_episode_steps,
                   tasks_to_complete=[task], render_mode=render_mode)
    env = HeldSetpointWrapper(env)
    env = VLAObservationWrapper(env, image_size=self.native_image_size)
    return ObsReshapeWrapper(env, image_size=self.image_size)
```

Let's see what one observation looks like once all three are on:

<div class="output"><p class="output-label">Printing each key's shape and dtype gives</p>

```text
camera_scene   (448, 448, 3)  uint8
joint_pos      (9,)  float32
joint_vel      (9,)  float32
camera_scene range   : 0 - 255
```
</div>

`max_episode_steps` is 400 because the longest human demonstration on file runs
to 314 steps, so a policy still going at 400 has failed by any reasonable
reading.

## The two lines the VLA changed

Everything above arrived working. Here is the entire environment-side difference
between the convolutional policy and ours.

`ObsReshapeWrapper` used to finish by transposing to channels-first and declaring
its space to match, because that is what `Conv2d` consumes:

```python
reduced = frames.resize(observation["camera_scene"], self.image_size)
return {**observation, "camera_scene": reduced.transpose(2, 0, 1)}
# and:  spaces.Box(0, 255, (3, image_size, image_size), np.uint8)
```

Our model permutes to channels-first on the GPU inside `preprocess`, which
Chapter 9 covers, so a transpose here would only be undone a moment later. Drop
it, and rewrite the `Box` shape to `(image_size, image_size, 3)`. That's it.

<div class="trap">
<span class="note-label">Trap · both sides of the line must agree</span>
What actually matters is not which order you pick but that the dataset and the
environment pick the same one. They disagreed exactly once, when
<code>sample_batch</code> dropped its transpose and this wrapper did not, and the
rollout died with <code>expected input to have 3 channels, but got 448</code>.
That was the lucky outcome. Had the numbers happened to line up, training would
have run to completion on transposed pixels and produced a quietly worse policy
with nothing anywhere reporting a problem.
</div>

Next, we'll look at the dataset, which needed one line changed for the same
reason, and which contains two facts about the demonstrations that will change
the loss function we write in Chapter 10.
