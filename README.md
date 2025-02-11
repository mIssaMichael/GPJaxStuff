# GPs and Stuff in Jax 
Currently, the repository is named Bayes-Jax, but I'm considering a more descriptive name: GPs-and-Stuff-in-Jax. (Those familiar with the MATLAB ecosystem—bless your heart—might recognize the blatant ripoff of [GPstuff](https://github.com/gpstuff-dev/gpstuff).)

The GP part covers the standard functionalities of exact inference for Gaussian Processes (GPs) with standard kernels (prior in "Bayesglish") and likelihoods. The Stuff part can include anything—think Bayesian GPs with Latent Variables, Hilbert Space Approximations, Deep GPs, Gaussian Markov Random Fields, Integrated Nested Laplace Approximations, Generalized GPs (adding a link function basiscally), multi-output GPs (basically single-output GPs in a trenchcoat).

Something I'm considering is finding a way to integrate the flexibility of fitting arbitrary models in Bayeux with specifying arbitrary GP-like densities densities for them (similar to what Dynamax does). Tentatively, this is all a playground for experimentation and learning. Hopefully, something good will come out of it (unlikely).
