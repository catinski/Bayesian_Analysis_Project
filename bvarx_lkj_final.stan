data {
  int<lower=1> T;
  int<lower=1> K;
  int<lower=1> M;
  matrix[T, K] X;
  vector[M] y[T];
}

parameters {
  matrix[K, M] z_beta;
  real<lower=0> tau;
  vector<lower=0>[M] sigma;
  cholesky_factor_corr[M] L_Omega;
}

transformed parameters {
  matrix[K, M] beta;
  matrix[M, M] L_Sigma;

  beta = z_beta * tau;
  L_Sigma = diag_pre_multiply(sigma, L_Omega);
}

model {
  tau ~ normal(0, 1);
  to_vector(z_beta) ~ normal(0, 1);
  sigma ~ exponential(1);
  L_Omega ~ lkj_corr_cholesky(2);

  for (t in 1:T) {
    y[t] ~ multi_normal_cholesky(X[t] * beta, L_Sigma);
  }
}

generated quantities {
  corr_matrix[M] Omega;
  cov_matrix[M] Sigma;

  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, sigma);
}
