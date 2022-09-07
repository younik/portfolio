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


For render modes that are expected to return something, you can still get the result with `.render()`:
{% highlight python linenos %}
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

env.reset()

for _ in range(100):
    env.step(env.action_space.sample())
    env.render()  # return a ndarray frame

env.close()
{% endhighlight %}

With this new API, we also introduce `_list` modes. In this case, the environment will save all the frames internally, and you can retrieve all of them at once calling `render`:

{% highlight python linenos %}
env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")

env.reset()

for _ in range(100):
    env.step(env.action_space.sample())

frame_collection = env.render()
env.close()
{% endhighlight %}
The frame collection will be a list of 101 *rgb arrays* (1 after the initial reset and 100 steps).

If you don't want rendering, you can just ignore the `render_mode` attribute.


## How to update your environment?

Updating your environment to this new API is super easy. \\
You just need to follow these 3 steps: 

**1. Add render_mode argument to init**:
```diff

- def __init__(self, ...)
+ def __init__(self, render_mode: Optional[str] = None, ...)
    ...
+   self.render_mode = render_mode
```

We assume that every environment can not render; thus, you don't need to add *None* to `self.metadata["render_modes"]`. Remember to specify the attribute `render_mode` for your environment. Then:

**2. Remove mode argument to render function and change all occurrences**
```diff
- def render(self, mode="human"):
+ def render(self):
+   mode = self.render_mode

```

For human mode, we want rendering to be automatic, thus:

**3. Update the step and reset methods**
```diff
def step(self, action):
    ...
+   if self.render_mode == "human":
+       self.render()

...

def reset(self, ...):
    ...
+   if self.render_mode == "human":
+       self.render()
```

At this point, your environment is supporting the new render API. 
Your environment will automatically support also collection of frames because this is handled by [RenderCollection](https://github.com/openai/gym/blob/master/gym/wrappers/render_collection.py) wrapper.
If you want a custom code for handling frame collection modes, you just need to add the new mode to `env.metadata['render_modes']`, and  the environment will not be wrapped by `RenderCollection`.

## Why this API?

As you may noticed, this API forbids the change of the render mode on-the-fly, but let the environment knows the render mode at initialization. This is important for some environment, since they need different initialization process for different mode, thus overcomplicated code to adhere to the old API. Moreover, some environments don't naturally render at each step (or they have multiple frames if they aren't static) but easily generate the rendering at the end of the episode. Finally, old API didn't allow to extract a smooth video when using *frame skipping*.