### Loss function

$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

### Theoretical guarantee

$\min\limits_{G} V(D, G)$ is achieved if and only if the probability distribution of the generator matches that of the real data.

### Proof

First let's prove that

For a static generator, the maximum of the loss function with respect to the discriminator is a constant:

$$V(D^*, G) = -\log(4)$$

Expand loss function to integral form

$$V(G, D)=\int_{\boldsymbol{x}} p_{\text {data }}(\boldsymbol{x}) \log (D(\boldsymbol{x})) d x+\int_{z} p_{\boldsymbol{z}}(\boldsymbol{z}) \log (1-D(g(\boldsymbol{z}))) d z$$

Make a change of term in our second term so that we integrale over generator's output G(z)$ directly.

$$\int_{\boldsymbol{x}} p_{\text {data }}(\boldsymbol{x}) \log (D(\boldsymbol{x}))+p_{g}(\boldsymbol{x}) \log (1-D(\boldsymbol{x})) d x$$

Maximize its value at every single point, or maximize

$$ a \log (y)+b \log (1-y)$$

Solve this we will get $y = \frac{a}{a+b}$, or $D(x)^{*}=\frac{p_{\text {data }}(x)}{p_{\text {data }}(x)+p_{g}(x)}$

A sidenote: we can see that if the probability distribution of the real data and the generated data are identical, then the optimal discriminator just makes random guess. This makes intuitive sense.

Therefore, for the generator, the upper bound of loss is:

$$
\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log \frac{p_{\text {data }}(\boldsymbol{x})}{P_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \frac{p_{g}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]
$$

And recall the definition of KL-divergence and JS-divergence:

$$
\begin{aligned}
D_{K L}(P \| Q) &=\mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right] \\
&=\mathbb{E}_{x \sim P}\left[\log \frac{2 P(x)}{2 Q(x)}\right] \\
&=\mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x) / 2}\right]-\log (2)
\end{aligned}
$$

$$
J S D(P \| Q)=\frac{1}{2} D_{K L}\left(P \| \frac{P+Q}{2}\right)+\frac{1}{2} D_{K L}\left(Q \| \frac{P+Q}{2}\right) \\
$$

So we can rewrite out loss function:

$$
\begin{aligned}
V(D^*, G)
&=-\log (4)+K L\left(p_{\text {data }} \| \frac{p_{\text {data }}+p_{g}}{2}\right)+K L\left(p_{g} \| \frac{p_{\text {data }}+p_{g}}{2}\right) \\
&=-\log (4)+2 \cdot J S D\left(p_{\text {data }} \| p_{g}\right)
\end{aligned}
$$

When the two probability distributions are equal, JS divergence is 0 and loss function is minimized, and its value is $-\log(4)$.

In actual training process, we first optimize the discirminator to convergence by locking the generator, and then update the generator. This is mimicing the mathamatical process discussed above.
