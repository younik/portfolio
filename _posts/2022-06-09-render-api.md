---
layout: post
title: New render API for gym library
date: 2022-06-09 21:40:16
description: The new Render API for gym is out!
tags: gym
categories: programming AI
---

I am writing this blog post to explain what has changed, the reasons behind this and how to adapt your current environment to the new API.

## What is new?

If you have ever used Gym, you are probably familiar with this code:


{% highlight python linenos %}
import gym

env = gym.make("FrozenLake-v1")
env.reset()

for _ in range(100):
    env.step(env.action_space.sample())
    env.render()

env.close()
{% endhighlight %}

where we just create an instance of the *FrozenLake* environment and we act randomly for 100 steps. In line 8, we render a single frame representing the current state of the environment. The method `render` accepts a keyword argument *mode*; in the above case, we use the default value, i.e. *human*.

With the new API, you need to specify the render mode at initialization, and the environment will take care to render each frame. Thus, in the case of *human* rendering, you don't need to call the `render` at all!

{% highlight python linenos %}
env = gym.make("FrozenLake-v1", render_mode="human")

env.reset()

for _ in range(100):
    env.step(env.action_space.sample())

env.close()
{% endhighlight %}

Ok, but **how does it work for non-human rendering?**
The environment will save all the frames internally, and you can retrieve all of them at once, calling `render`:

{% highlight python linenos %}
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

env.reset()

for _ in range(100):
    env.step(env.action_space.sample())

frame_collection = env.render()
env.close()
{% endhighlight %}
The frame collection will be a list of 101 *rgb arrays* (1 after the initial reset and 100 steps).

If you don't want rendering, you can just ignore the `render_mode` attribute; if you need to render just some frames, and the environment allows it, you can use the special mode `single_rgb_array`. In this case, the method `render` will return a single frame, as before.


## How to update your environment?

Updating your environment to this new API is super easy. \\
You just need to follow these 5 steps: 

**1. Rename your current render function as *_render_frame* (for example)**
```diff
- def render(mode: str = "human")
+ def _render_frame(mode: str = "human")
```

**2. Add render_mode in init and use the Renderer utility class**:
```diff
+ from gym.utils.renderer import Renderer
...

- def __init__(self, ...)
+ def __init__(self, render_mode: Optional[str] = None, ...)
    ...
+   assert render_mode is None or render_mode in self.metadata["render_modes"]
+   self.render_mode = render_mode
+   self._renderer = Renderer(self.render_mode, self._render_frame)
```

We assume that every environment can not render; thus, you don't need to add *None* to `self.metadata["render_modes"]`. Remember to specify the attribute `render_mode` for your environment. The renderer will collect a frame at each step using the function we passed, thus:

**3. Update the step and reset methods**
```diff
def step(self, action):
    ...
+   self._renderer.render_step()

...

def reset(self, ...):
    ...
+   self._renderer.reset()
+   self._renderer.render_step()
```


**4. Write the actual render function**
```diff
+ def render():
+   return self._renderer.get_renders()
```

If you want to maintain backward compatibility, you will probably need to write something like this:
```diff
+ def render(self, mode: str = "human"):
+   if self.render_mode is not None:
+      return self._renderer.get_renders()
+   else:
+      return self._render_frame(mode)
```
In this case, remember to remove the *mode* argument along with the release of Gym 1.0, since it will be removed.
At this point, your environment is supporting the new render API. However, your environment is only supporting collections of frames, thus it is not handy for rendering just part of episodes. You should

**5. Support single-frame rendering**

Most of the time, it just means to add *single_rgb_array* to your `render_modes`:

```diff
- metadata = {"render_modes": ["human", "rgb_array"], ...}
+ metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], ...}
```

and update your `_render_frame` function to behave equally for *rgb_array* and *single_rgb_array*.
For example, replace if statements in this way:
```diff
- if mode == "rgb_array":
+ if mode in {"rgb_array", "single_rgb_array"}:
```

If you have non-standard rendering modes, you can customize the Renderer class; visit the [documentation](https://github.com/openai/gym/blob/master/gym/utils/renderer.py) for more information.

For more a complicated environment, the process can be different; if you need any help **feel free to contact me.** 

## Why this API?

As you may noticed, this API gives more freedom for the rendering behavior of the environment. On the other side, it forbids the change of the render mode on-the-fly, but let the environment knows the render mode at initialization. This is important for some environment, since they need different initialization process for different mode, thus overcomplicated code to adhere to the old API. Moreover, some environments don't naturally render at each step (or they have multiple frames if they aren't static) but easily generate the rendering at the end of the episode. Finally, old API didn't allow to extract a smooth video when using *frame skipping*.