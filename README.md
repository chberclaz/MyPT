# MyPT

First interation of a LLM concept done after a tutorial by Andrej Karpathy (https://www.youtube.com/@AndrejKarpathy) (nanoGPT on Github)
He has a whole youtube series dedicated to machine Learning and on how to code them (not just implement existing Models)

As i try to learn these toppics this is as good a starting point as any other...

wish me luck and let's see where this gets me :)

Goal: What is the Neural Network under the Hood, that models the sequenze of these words. The Model is derived from the Paper "Attention is all you need Dec 2007 Long Beach USA" (Credits found in Andrejs Videos)
![alt text](image.png)

Train a transformer based language Model. In our Case train a character level Model. As Training Set we will use Tiny Shakespear (a concatination of all of shakespear; ca 1MB)
We will train our Transformer to write text in the same style as shakespear.

GPT Model should be only 2 Files at around 300 Lines of Code
1 File defines the Model
1 File trains it on some given text
