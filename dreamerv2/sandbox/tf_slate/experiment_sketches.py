"""
Goal: is to remove the bump in the reward_loss.

Observation: It seems like this bump is correlated with the dip in the dvae loss. 

Hypothesis: maybe this observation is caused by the fact that we are using discrete latents.

To test:
- discrete hard input (yes/no)
- discrete hard output (yes/no)
- stop gradient input (yes/no)
- stop gradient output (yes/no)
- continuous 

input: discrete vs continuous
    if discrete: 
        hard vs soft
        stop gradient input (yes/no)
output: discrete vs continous
    if discrete: 
        hard vs soft
        stop gradient output (yes/no)



Steps:
- have an end-to-end training pipeline: do the entire forward pass all at once with two gradient tapes, then backprop with separate optimizers from the same loss: one forward pass, shared loss.
- within discrete:
    flag for hard vs soft in input
    flag for hard vs soft in output
    flag for stop gradients in input
    flag for stop gradients in output
- then you might need to write a different network architecture for continuous. Try to make as few modifications as possible. What would that look like? Essentially the things we'd need to change are:
    - output layer of dvae encoder
    - create_tokens
    - loss function of slot_model: you'd probably just want to just feed the output to the dvae decoder instead

Test on static first, but keep in mind that dynamic might change things. Look to see if it still differentiates slots on dmc properly

Ok, I just verified that monolithic_train_step has the same behavior as modular train step. So let's see if we can actually just quickly test some of these out. Oh hmmm, it seems like it's not actually calling monolithic_train_step?

Ok, can we quickly just test what happens if we do not stop gradient?
Let's see.

Create a method for handling stop gradient
- consider which situations you are creating tokens
- Hmm, why does the behavior not change when I remove the stop gradient? Oh interesting: it only changes if I remove the stop gradient to the target?
    sg_inp  |  sg_out  |  changed behavior?

    True       True        (default)
    True       False        yes, learns faster (about 1/3 due to dvae, 2/3 due to slot model)
    False      True         no, same as default
    False      False        same behavior as sg_inp=True, sg_out=False

Ok interesting. Why is it that stopping the gradient into the input does not change anything, but stopping the gradient changes something?

Let's ask if the grad norms have changed. (this is with mono train)

    sg_inp  |  sg_out  |  does grad norm change?

    True       True        (default)
    True       False       dvae grads change
    False      True        neither dvae nor sm, same as default
    False      False       dvae grads change

Would it error out if I tried to do modular_train instead of mono train?


    sg_inp  |  sg_out  |  does it complain if we train separately?

    True       True        (default)
    True       False        no, same behavior as mono train
    False      True         no, same behavior as mono train
    False      False        no, same behavior as mono train

Oh interesting: why doesn't it complain? So I certainly can see how not stopping the gradients into the output would affect slot_model, but why this would affect the dvae if we already applied the gradients to the dvae before we handle_stop_gradient in the modular train_step? ^IGNORE THIS ABOVE RESULT: it's because I still had mono_train=True

Trying again: Would it error out if I tried to do modular_train instead of mono train?


    sg_inp  |  sg_out  |  does it complain if we train separately?

    True       True        (default)
    True       False       does not complain, same behavior as default
    False      True        does not complain, same behavior as default
    False      False       does not complain, same behavior as default

Ah interesting: it seems like if we do modular train, then that automatically stops the gradients from affecting sm because it's a different gradient tape, and it automatically stops the gradients from affecting dvae because the gradients have already been applied before we called handle_stop_gradients.

Ok, now the question is: 
- why does changing sg_out affect the gradients of dvae and not sm?
    - why does changing sg_out affect the gradients of dvae?
        - because this would actually end up pull the dvae z_hard tokens towards the output of the sm. The z_target was produced by straight-through, so the sm output will contribute to the gradient of the y_soft in the gumbel_softmax
    - why does changing sg_out not affect the gradients of sm?
        - I suppose this is because in the cross entropy loss we are just multipliying the log_softmax(pred) by 1 or 0, so from sm's perspective, the fact that we did not stop gradient into 0 doesn't matter because we were zeroing out that gradient anyways. Also, the fact that we did not stop the gradient into 1 doesn't matter because we'd just be multiplying whatever gradient we get by 1 anyways
            - although we can test this if we make the z be soft instead
    - so it seems like if we want the gradients going into sm to change, we'd actually want to change hard to soft, at least between dvae.encoder and sm. --> need to test this
- why does changin sg_inp not affect dvae nor sm?
    - why does changing sg_in not affect the gradients of dvae?
        - is this mainly because I'm not implementing the straight-through gradient properly, in the sense that it's just blocking the gradient? --> No, I don't think so, because if I unblock the gradient going into the target, the dvae does change.
            - verify that if I remove the straight-through gradient, but unblock the gradient going into the target, the dvae will not change --> verified. The straight-through gradient does indeed pass gradients through the target if I unblock the gradients going into target.
        - why is it that the loss from the slot model does not propagate back through the z_input? Could it be because of the one-hot dictionary? --> it might be because of the one-hot dictionary
            - let's test it. In order to test this I might actually need to implement a different way of selecting the token_embs. I might instead of have to do an einsum thing. 
                - ah yes, if I do the einsum thing, then the gradients are indeed different. If I stop the gradient though, the dvae is not affected. Wait actually this is confusing: if I stop the gradient, then why should the gradient be different again, if I'm stopping the gradient? I think it's just a numerical error
    - why does changing sg_in not affect the gradients of sm?
        - because sm is downstream of of sg_in anyways
    - Note that z_input is computed from z_target, so I would assume not stopping gradients into the input would also affect z_target. So why doesn't the gradient leak into z_target?
        - so in the case of sg_inp=False and sg_out=False, it doesn't matter, because we aren't stopping the gradients going into z_target anyways
        - in the case of sg_inp=False and sg_out=True, this answer would be the same a the answer for "why does changing sg_out not affect the gradients of dvae?" because maybe the gradient does leak into z_target, but dvae would still be similarly unaffected


        what if we computed z_input first and then z_target?

Why is it that if I remove the straight through gradient it doesn't change the gradient going into dvae? I would have thought that that would block the gradient going into dvae?
    - oh it's because the dvae decoder does not train with hard=True


Stackpointer:
- it seems like the reason why sg_inp=True and sg_out=False did not create changes in the dvae is because the onehot dictionary cut off the gradients
- we are currently trying to understand what effect the einsum version of onehot dictionary has on the gradients going into the dvae. I know that if I use the old onehot dictionary, then the gradients of the dvae are not affected whether or not I stop the gradient to z_input. I also observe that if I use einsum onehot dictionary, then the gradients of the dvae are affected even though I stop gradient into both z_input and z_output! Why is this the case? My next question is to figure out why. --> it seems to mostly be a numerical error.

Ok, now what you can do is you can ask the question again when you use z_sample rather than z_hard as input. Oh wait hold on, you indeed are using self.sample. Now, given that we are using the onehot dictionary version now, let's see what happens here.

    sg_inp  |  sg_out  |  does grad norm change?

    True       True        (default)
    True       False       dvae changes
    False      True        dvae changes
    False      False       dvae changes

for all three options where dvae changes, they all change in different ways


Ok, now let's see what happens if we make z_hard be soft instead

    sg_inp  |  sg_out  |  does grad norm change?

    True       True        (default)
    True       False       dvae changes
    False      True        dvae changes
    False      False       dvae changes

given that z_hard is soft, why don't the sm gradients change?
    this is in the case of sg_out=False. Yeah, actually we wouldn't expect the gradient of sm to change because there is no computation downstream of sm that would influence sm.
    - we would expect the sm gradients to change if we made sm predict the input of the dvae decoder.

Ok, what experiments can you run right now? You can run the experiments of:
discrete, hard, but turn sg_inp and sg_out on and off
- do it for 3d_shapes to sanity check
- do it for balls to see how it affects performance
(what would have changed from before, even with default, would be (1) the new onehot dictionary (2) mono_train=True)


then: you will test what happens if you do soft and hard

then you will test what happens if you hook up the slot model output as decoder input
- discrete
    - hard vs soft
    - stop gradient (yes/no)
- continuous
"""