data {
  int<lower=2> T;
  vector[T] y;      // Log-prices
}

parameters {
  real mu;             // Il parametro Drift
  real<lower=0> sigma; // Volatilità
}

model {  
  mu ~ normal(0, 0.01); 
  sigma ~ exponential(1);

  // LIKELIHOOD
  for (t in 2:T) {
    y[t] ~ normal(y[t - 1] + mu, sigma);
  }
}

generated quantities {
  vector[T - 1] log_lik;
  real y_next;

  for (t in 2:T) {
    // Calcolo della log-likelihood per il confronto dei modelli (LOO-CV)
    log_lik[t - 1] = normal_lpdf(y[t] | y[t - 1] + mu, sigma);
  }

  // Forecast per il periodo T+1
  // Il valore predetto sarà l'ultimo prezzo osservato + drift + rumore
  y_next = normal_rng(y[T] + mu, sigma);
}
