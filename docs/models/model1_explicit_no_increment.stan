data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real alpha_unc; 
    real beta_unc;
    real sigma_unc;
}

transformed parameters {
    real alpha = exp(alpha_unc); 
    real beta = exp(beta_unc);   
    real sigma = exp(sigma_unc); 
}

model {
    y ~ normal(alpha + beta * x, sigma);
}