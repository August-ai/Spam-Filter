<h1>Spam filter using a Naive Bayes model</h1>

<p>The model gives a probability that a message is a normal sms vs a SPAM given a prior knowledge of the world, by using the following formulas:</p>


<img style="-webkit-user-select: none;margin: auto;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;" src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathbf{P(Spam%20|%20w_1,w_2,%20...,%20w_n)%20\propto%20P(Spam)%20\cdot%20\prod_{i=1}^{n}P(w_i|Spam)}}">

<img style="-webkit-user-select: none;margin: auto;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;" src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathbf{P(Ham%20|%20w_1,w_2,%20...,%20w_n)%20\propto%20P(Ham)%20\cdot%20\prod_{i=1}^{n}P(w_i|Ham)}}">
</br>
</br>
<p>Initially by https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html</p>
