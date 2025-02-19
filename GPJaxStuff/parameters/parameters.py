# Parameters should be any thing that can be a suitable type in a pytree object
# Question: How would we copy the handling of adding adjsutments to the log density if we're doing Bayesian GPs? Do we want to automatically handle it as the parameters block does in Stan? Seems like maybe we just give people the option. Maybe we do a Stan-like interface depedning on the model. Currenly we don't have to build in anything that seperates implicit from explicit contributions to calculating the log density or any features of the model. Allow the user to specify if they need a correction. We do the correction only if that specific type of model requires it.
# Pretty much I want to do this:
# params = {
#       'a': Real(jnp.array([1.0], lower=0)),
#       'b': Real(2.0, lower=3, upper=4)),
#       'c': Int(2.0, upper=5)
#       'd': LowerTriangular(jnp.array([[1.0, 0.0], [0.5, 1.0]])),
#       'e': Static(3.33),
# }
