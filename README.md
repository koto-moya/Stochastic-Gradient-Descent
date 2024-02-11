# Implementing SGD

**Using SGD to solve for the speed of a roller coaster**

Allowing ourselves to guess that the speed of the roller coaster can be modeled by a qudratic function, it greatly simplifies the work we have to do.  

Now all we need to do is adjust the parameter of a, b, and c to find the equation for a given set of inputs

In other words, we need to finds the *best* quadrictic function that fits the data

**Loss** will determine the best quadratic, and we will use *Mean Squared Error*


## 7 steps to implementing SGD

### Step 1: Init the parameters

Initialize to random values and ensure that gradients are turned on for them. 

Create a rank 1 tensor with 3 values. 

### Step 2: Make predictions

Calculate the predictiosn given your init parameters

> to graph you need to use .detach() on predictions to get the tensor values since grad is applied

### Step 3: Calculate the loss

Use the Mean Squared Error to find the loss

### Step 4: Calculate Gradients

Apply backwards to the loss
Then check the grad on the params

> damn Pytorch takes care of everything for you

### Step 5: Step up the weights

Decide the learning rate
apply the learning rate of the params
set the grad to None to restart the process

> To access the raw values of the grad use the .grad.data function
> Hard to see in one step so create a function that does all of this for you and iterate through a loop

### Step 6: 

Repeat step 2 through 5 in a loop (steps 2-5 can be made into a function)


### Step 7: 

Stop the learning
This can be an arbitrary decision but can also be based on some hueristic

## Applying What I've learned

I applied the 7 steps to apporximate a cos function.  Really simple, just to moitivate the intuition for the technique.

Used the Mclaurin approximation:  $cos(x) \approx a_0 - a_1\frac{x^2}{2!}+a_2\frac{x^4}{4!}-a_3\frac{x^6}{6!}+a_4\frac{x^8}{8!}$

Where the $a_0 - a_4$ are the parameters (weights) with which I apply the gradient adjustments

ended up using the 7th approx cause why not

Used the torch.cos as the target. Tried both with noise and without and got good results

Again this is not about the best cos approximator (I would just get 1 mn GPUs and run the biggest mclaurin anyone has ever seen lol) just building intuition on SGD

The way I see it, the target is some n-dim surface (using that term loosely here), the predictions are points along a blanket you are laying over the top of the surface.  Then you push all of the spots where voids are felt. You push with a force determined by the gradient of the void and the importance you give that gradient (learning rate). Theoretically, this process could fill in all of the nooks and crannies of that surface arbitrarily over a long enough period.  In practice this would lead to over-fitting of noisy data and take a really long time in some cases.  This is where my analogy falls apart.  I could make a more contrived version by saying that your blanket is a special one that (should) settles on the gestalt of your surface, but who likes analogies anyway?  




