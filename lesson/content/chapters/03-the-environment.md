---
title: "The environment"
part: "Part II · The pieces"
chapter: 3
weight: 3
standfirst: "What one observation is, and three wrappers that make the arm behave."
---

Almost everything in this chapter came over from the behavior-cloning project
unchanged. We're reading it rather than writing it, because you cannot reason
about the chapters that follow without knowing exactly what one observation is
and what the environment does with the numbers you send back.

There is one change the VLA needs, and it's two lines. We'll get to it at the
end.

## What the environment gives you

Franka Kitchen is a MuJoCo simulation: a 7-joint Franka arm with a 2-finger
gripper, in a kitchen with a microwave, a kettle, two cabinets, a light switch
and four burner knobs.

```python
env = gym.make("FrankaKitchen-v1", max_episode_steps=400,
               tasks_to_complete=["hinge cabinet"], render_mode="rgb_array")
```

The raw observation is a flat vector mixing joint state, object poses, and
goal information. You do not want most of it. A VLA reads the scene from
pixels and the goal from language, so object poses and goal vectors are
redundant by construction.

**The action space is nine numbers in `[−1, 1]`.** Seven arm joint velocities
and two gripper commands. This matters more than it looks, and it is worth
reading the environment source to confirm rather than assuming: the
environment integrates your action into a position setpoint itself. You are
already in a normalized velocity space.

<div class="checkpoint">
<span class="note-label">Why this decision comes first</span>
Most VLA pipelines quantile-normalize the action space. Here that would be a
second normalization on top of one the environment already did. Five minutes
reading <code>kitchen_env.py</code> removed a component from the design. Read
your environment's contract before designing around it.
</div>

## Wrapper one: stop the arm sagging

The first problem has nothing to do with learning. Send zero action for 200
steps and the arm droops about 61 degrees.

The environment integrates each action onto the *measured* joint position
every step. Gravity pulls each joint slightly below its commanded setpoint,
that droop is read back in, and the next setpoint starts from the drooped
position. The arm ratchets downward.

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

Holding the setpoint you commanded turns 61 degrees of drift into about 1
degree of offset, which is what a position servo should do.

`max_lead` caps how far the setpoint may run ahead of the arm. Without it the
setpoint outruns the joints during a fast move and they coast for about 20
steps after you release the stick. It has to stay above the servo's own
steady-state droop of roughly 0.02 radians, or the clamp re-anchors to
measured position and the sag returns. At 0.05: sag 1.4 degrees, coast 2.9
degrees.

## Wrapper two: build the observation you want

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

Three keys. The camera view is the free camera from the environment's default
config — the same view a human demonstrator sees, so demonstrator and policy
look at the same thing.

Object poses and goal vectors are dropped. They were the only reason
observation width varied by task (61, 63 or 73 depending on the task), which
made it impossible to mix demonstrations from different tasks in one buffer.

<div class="note">
<span class="note-label">Note</span>
<code>joint_pos</code> and <code>joint_vel</code> are carried through the
observation but are <strong>not fed to the model</strong> in this build.
<a href="{{< relref "chapters/15-where-this-goes" >}}">Chapter 15</a> explains why that omission is
deliberate, and when to reverse it.
</div>

## Wrapper three: match the training resolution

Frames are archived at 896 pixels so recorded demonstrations stay useful to a
model that wants more pixels later. Training reduces them to 448 on load. The
live environment has to apply the identical reduction, or a rollout sees
something slightly different from what training saw.

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

**Frames come out HWC** — height, width, channels. The dataset uses that same
order.

This is the one place in the whole environment layer that the VLA changed. The
conv policy wanted channels-first, so the wrapper used to end with
`reduced.transpose(2, 0, 1)` and declare its `Box` as
`(3, image_size, image_size)`. Our model permutes to channels-first on the GPU
inside `preprocess`, which Chapter 9 covers, so a transpose here would only be
undone a moment later. Two lines: drop the transpose, and rewrite the `Box`
shape. Keeping both sides identical is the entire job of this wrapper,
and [Chapter 14]({{< relref "chapters/14-traps" >}}) covers what happened the one time they
disagreed.

## Assembling them

<p class="filename">Filename: <strong>agent.py</strong></p>

```python
def _make_env(self, task, render_mode):
    env = gym.make("FrankaKitchen-v1", max_episode_steps=self.max_episode_steps,
                   tasks_to_complete=[task], render_mode=render_mode)
    env = HeldSetpointWrapper(env)
    env = VLAObservationWrapper(env, image_size=self.native_image_size)
    return ObsReshapeWrapper(env, image_size=self.image_size)
```

Order matters. `ObsReshapeWrapper` is outermost so it reshapes whatever the
layer below produced. Anything that wants full-resolution frames — recording,
debugging — goes below it.

`max_episode_steps` is 400. The longest human demonstration on file is 314
steps, so a policy still going at 400 has failed.
