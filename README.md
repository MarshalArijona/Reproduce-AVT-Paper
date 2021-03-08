# CLUB-Generative-Network
# Abstract
Abstract—This research is about the application of mutual
information estimator for generative models. This research applies variational contrastive log-ratio upper bound (vCLUB)
minimization algorithm to minimize the mutual information
between mixture distribution between real data distribution and
generated data distribution and binary distribution that alternate
between real data distribution and generated data distribution.
The aim is the same as minimizing Jensen-Shannon divergence
between real data distribution and generated data distribution
which is the purpose of generative adversarial network. Furthermore, this research proposed two MI-based generative models,
CLUB-sampling generative network (vCLUB-sampling GN) and
vCLUB-non sampling generative network (vCLUB-non sampling
GN). Both models are developed as deep neural networks. Result
of experiments show that vCLUB-non sampling generate better
samples than vCLUB-non sampling GN and variational L1-out
generative network (vL1-out GN). Unfortunately, both vCLUB
GN models are outperformed by generative adversarial network
(GAN).
# Algorithm
\begin{algorithm}[H]
\caption{Training algorithm for vCLUB-GN}\label{alg:vclub}
\begin{algorithmic}
\FOR{each training iteration}
    \STATE - Sampling $\{(\boldsymbol{x_{i}}^{(data)}, \boldsymbol{y_{i}})\}_{i=1}^{N}$ from $p_{data}(\boldsymbol{x})$.
    \STATE - $\mathcal{L}_{data}(\theta) = \frac{1}{N} \sum_{i = 1}^{N} \log q_{\theta}(y_{i} | x_{i}^{(data)})$
    \STATE - Sampling $\{(\boldsymbol{x_{i}}^{(g)}, \boldsymbol{y_{i}})\}_{i=1}^{N}$ from $p_{g}(\boldsymbol{x})$.
    \STATE - $\mathcal{L}_{g}(\theta) = \frac{1}{N} \sum_{i = 1}^{N} \log q_{\theta}(y_{i} | x_{i}^{(g)})$
    \STATE - Update $q_{\theta}(y | x)$ by maximizing $\mathcal{L}_{data}(\theta) + \mathcal{L}_{g}(\theta)$
    \FOR {$i=1$ to $N$}
        \IF{use $sampling$}
            \STATE $U_{i} = \log q_{\theta}(y_{i} | x_{i}^{(g)}) - \log(1 - q_{\theta}(y_{i} | x_{i}^{(g)}))$
        \ELSE
            \STATE $U_{i} = \log q_{\theta}(y_{i} | x_{i}^{(g)}) - \frac{1}{2} (\log(1 - q_{\theta}(y_{i} | x_{i}^{(g)})) + \log q_{\theta}(y_{i} | x_{i}^{(g)}))$
        \ENDIF
    \ENDFOR
    \STATE - Update $p_{g}$ by minimizing $I_{vCLUB} = \frac{1}{N} \sum_{i = 1}^{N} U_{i}$
\ENDFOR
\end{algorithmic}
\end{algorithm}
